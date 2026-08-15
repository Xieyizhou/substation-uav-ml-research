import Foundation

public struct RuntimeCompatibilityManager: @unchecked Sendable {
    public let store: RuntimeProfileStore
    private let inspector: RuntimeInspector
    private let fileManager: FileManager
    private let environment: [String: String]

    public init(
        store: RuntimeProfileStore = RuntimeProfileStore(),
        inspector: RuntimeInspector = RuntimeInspector(),
        fileManager: FileManager = .default,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        self.store = store
        self.inspector = inspector
        self.fileManager = fileManager
        self.environment = environment
    }

    public func assess(
        projectRoot: URL,
        profile: SandboxProfile,
        selection: RuntimeSelection = RuntimeSelection(),
        enforceSavedIdentity: Bool = true
    ) throws -> RuntimeAssessment {
        let root = try ProjectLocator.validate(root: projectRoot, fileManager: fileManager)
        let manifest = try RuntimeCompatibilityManifest.load(projectRoot: root)
        let saved = store.load(fileManager: fileManager).flatMap {
            URL(fileURLWithPath: $0.projectRoot).standardizedFileURL == root ? $0 : nil
        }
        let python = inspector.python(
            candidates: pythonCandidates(root, selection, saved),
            manifest: manifest, root: root, profile: profile
        )
        let px4 = inspector.px4(
            candidates: px4Candidates(selection, saved), manifest: manifest
        )
        let gazebo = inspector.gazebo(
            candidates: gazeboCandidates(selection, saved),
            manifest: manifest, root: root, profile: profile
        )
        var formulaEnvironment = environment
        if let saved {
            formulaEnvironment["UAV_SANDBOX_OPENCV_PREFIX"] = saved.openCVPrefix
            formulaEnvironment["UAV_SANDBOX_QT_PREFIX"] = saved.qtPrefix
        }
        let opencv = inspector.formula(
            id: "opencv", title: "OpenCV", rule: manifest.opencv,
            environment: formulaEnvironment, root: root
        )
        let qt = inspector.formula(
            id: "qt", title: "Qt", rule: manifest.qt,
            environment: formulaEnvironment, root: root
        )
        var components = [python, px4, gazebo, opencv, qt]
        if enforceSavedIdentity, let saved,
           let change = identityChange(saved, components: components) {
            components.insert(change, at: 0)
        }
        return assessment(root: root, profile: profile, components: components)
    }

    public func validateAndPersist(
        projectRoot: URL,
        profile: SandboxProfile,
        selection: RuntimeSelection = RuntimeSelection()
    ) throws -> RuntimeAssessment {
        let value = try assess(
            projectRoot: projectRoot, profile: profile,
            selection: selection, enforceSavedIdentity: false
        )
        if let selected = value.selected { try store.save(selected, fileManager: fileManager) }
        return value
    }

    private func assessment(
        root: URL, profile: SandboxProfile, components: [RuntimeComponent]
    ) -> RuntimeAssessment {
        let blockers = components.filter { $0.status.blocks(profile) }.map {
            "\($0.title): \($0.detail)"
        }
        let warned = components.contains {
            $0.status == .untested || $0.status == .compatibleWithWarning
        }
        let overall: RuntimeCompatibilityStatus = blockers.isEmpty
            ? (warned ? .compatibleWithWarning : .compatible)
            : components.contains { $0.status == .changedSinceValidation }
                ? .changedSinceValidation
                : components.contains { $0.status == .missing }
                    ? .missing : .unsupported
        let selected = blockers.isEmpty
            ? buildProfile(root: root, components: components, overall: overall) : nil
        return RuntimeAssessment(
            components: components, overall: overall,
            blockers: blockers, selected: selected
        )
    }

    private func buildProfile(
        root: URL, components: [RuntimeComponent],
        overall: RuntimeCompatibilityStatus
    ) -> VerifiedRuntimeProfile? {
        let values = Dictionary(uniqueKeysWithValues: components.map { ($0.id, $0) })
        guard let python = values["python"], let pythonPath = python.path,
              let pythonVersion = python.version,
              let px4 = values["px4"], let px4Root = px4.path,
              let px4Commit = px4.identity,
              let gazebo = values["gazebo"], let gazeboPath = gazebo.path,
              let gazeboVersion = gazebo.version,
              let opencv = values["opencv"], let opencvPath = opencv.path,
              let opencvVersion = opencv.version,
              let qt = values["qt"], let qtPath = qt.path,
              let qtVersion = qt.version else { return nil }
        return VerifiedRuntimeProfile(
            schemaVersion: 1, projectRoot: root.path,
            pythonExecutable: pythonPath, pythonVersion: pythonVersion,
            px4Root: px4Root, px4Commit: px4Commit,
            gazeboExecutable: gazeboPath, gazeboVersion: gazeboVersion,
            gazeboDistribution: gazebo.detail.split(separator: "/").first.map {
                $0.trimmingCharacters(in: .whitespaces)
            } ?? "unknown",
            openCVPrefix: opencvPath, openCVVersion: opencvVersion,
            qtPrefix: qtPath, qtVersion: qtVersion,
            validationTimestamp: Date(), compatibilityResult: overall
        )
    }

    private func identityChange(
        _ saved: VerifiedRuntimeProfile, components: [RuntimeComponent]
    ) -> RuntimeComponent? {
        let values = Dictionary(uniqueKeysWithValues: components.map { ($0.id, $0) })
        let changes = [
            values["python"]?.path != saved.pythonExecutable
                || values["python"]?.version != saved.pythonVersion ? "Python" : nil,
            values["px4"]?.path != saved.px4Root
                || values["px4"]?.identity != saved.px4Commit ? "PX4" : nil,
            values["gazebo"]?.path != saved.gazeboExecutable
                || values["gazebo"]?.version != saved.gazeboVersion ? "Gazebo" : nil,
            values["opencv"]?.path != saved.openCVPrefix
                || values["opencv"]?.version != saved.openCVVersion ? "OpenCV" : nil,
            values["qt"]?.path != saved.qtPrefix
                || values["qt"]?.version != saved.qtVersion ? "Qt" : nil,
        ].compactMap { $0 }
        guard !changes.isEmpty else { return nil }
        return RuntimeComponent(
            id: "runtime_profile", title: "Verified runtime",
            status: .changedSinceValidation,
            path: store.fileURL.path,
            detail: "Runtime changed: \(changes.joined(separator: ", ")). Revalidation is required."
        )
    }

    private func pythonCandidates(
        _ root: URL, _ selection: RuntimeSelection,
        _ saved: VerifiedRuntimeProfile?
    ) -> [URL] {
        var values = [root.appendingPathComponent(".venv/bin/python")]
        values += [selection.pythonExecutable].compactMap { $0 }
        values += [saved.map { URL(fileURLWithPath: $0.pythonExecutable) }].compactMap { $0 }
        values += inspector.executableCandidates("python3", environment: environment)
        values += inspector.executableCandidates("python", environment: environment)
        return values
    }

    private func px4Candidates(
        _ selection: RuntimeSelection, _ saved: VerifiedRuntimeProfile?
    ) -> [URL] {
        [selection.px4Root,
         environment["PX4_ROOT"].map(URL.init(fileURLWithPath:)),
         saved.map { URL(fileURLWithPath: $0.px4Root) },
         fileManager.homeDirectoryForCurrentUser.appendingPathComponent("PX4-Autopilot")]
            .compactMap { $0 }
    }

    private func gazeboCandidates(
        _ selection: RuntimeSelection, _ saved: VerifiedRuntimeProfile?
    ) -> [URL] {
        var values = [selection.gazeboExecutable].compactMap { $0 }
        values += [saved.map { URL(fileURLWithPath: $0.gazeboExecutable) }].compactMap { $0 }
        values += [environment["UAV_SANDBOX_GZ_EXECUTABLE"].map(URL.init(fileURLWithPath:))]
            .compactMap { $0 }
        values += inspector.executableCandidates("gz", environment: environment)
        return values
    }
}
