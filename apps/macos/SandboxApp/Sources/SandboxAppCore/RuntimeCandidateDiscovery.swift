import Foundation

extension RuntimeCompatibilityManager {
    public func discoverCandidates(
        projectRoot: URL,
        profile: SandboxProfile,
        selection: RuntimeSelection = RuntimeSelection(),
        assessment: RuntimeAssessment? = nil
    ) throws -> [RuntimeCandidateGroup] {
        let root = try ProjectLocator.validate(root: projectRoot, fileManager: fileManager)
        let manifest = try RuntimeCompatibilityManifest.load(projectRoot: root)
        let saved = store.load(fileManager: fileManager).flatMap {
            URL(fileURLWithPath: $0.projectRoot).standardizedFileURL == root ? $0 : nil
        }
        let value = try assessment ?? assess(
            projectRoot: root, profile: profile, selection: selection
        )
        let selected = Dictionary(uniqueKeysWithValues: value.components.compactMap {
            item in item.path.map { (item.id, $0) }
        })
        return [
            pythonGroup(root, profile, manifest, selection, saved, selected),
            px4Group(manifest, selection, saved, selected),
            gazeboGroup(root, profile, manifest, selection, saved, selected),
            formulaGroup(
                id: "opencv", title: "OpenCV", rule: manifest.opencv,
                variable: "UAV_SANDBOX_OPENCV_PREFIX", manual: selection.openCVPrefix,
                saved: saved?.openCVPrefix, root: root, selected: selected
            ),
            formulaGroup(
                id: "qt", title: "Qt", rule: manifest.qt,
                variable: "UAV_SANDBOX_QT_PREFIX", manual: selection.qtPrefix,
                saved: saved?.qtPrefix, root: root, selected: selected
            ),
        ]
    }

    private func pythonGroup(
        _ root: URL, _ profile: SandboxProfile,
        _ manifest: RuntimeCompatibilityManifest,
        _ selection: RuntimeSelection, _ saved: VerifiedRuntimeProfile?,
        _ selected: [String: String]
    ) -> RuntimeCandidateGroup {
        var paths = [CandidatePath(root.appendingPathComponent(".venv/bin/python"), .repository)]
        append(selection.pythonExecutable, .manual, to: &paths)
        append(saved.map { URL(fileURLWithPath: $0.pythonExecutable) }, .saved, to: &paths)
        paths += inspector.executableCandidates("python3", environment: environment).map {
            CandidatePath($0, .path)
        }
        paths += inspector.executableCandidates("python", environment: environment).map {
            CandidatePath($0, .path)
        }
        return group("python", "Python", paths, selected) { path in
            inspector.python(
                candidates: [path], manifest: manifest, root: root, profile: profile
            )
        }
    }

    private func px4Group(
        _ manifest: RuntimeCompatibilityManifest,
        _ selection: RuntimeSelection, _ saved: VerifiedRuntimeProfile?,
        _ selected: [String: String]
    ) -> RuntimeCandidateGroup {
        var paths: [CandidatePath] = []
        append(selection.px4Root, .manual, to: &paths)
        append(environment["PX4_ROOT"].map(URL.init(fileURLWithPath:)), .environment, to: &paths)
        append(saved.map { URL(fileURLWithPath: $0.px4Root) }, .saved, to: &paths)
        paths.append(CandidatePath(
            fileManager.homeDirectoryForCurrentUser.appendingPathComponent("PX4-Autopilot"),
            .defaultLocation
        ))
        return group("px4", "PX4", paths, selected) { path in
            inspector.px4(candidates: [path], manifest: manifest)
        }
    }

    private func gazeboGroup(
        _ root: URL, _ profile: SandboxProfile,
        _ manifest: RuntimeCompatibilityManifest,
        _ selection: RuntimeSelection, _ saved: VerifiedRuntimeProfile?,
        _ selected: [String: String]
    ) -> RuntimeCandidateGroup {
        var paths: [CandidatePath] = []
        append(selection.gazeboExecutable, .manual, to: &paths)
        append(saved.map { URL(fileURLWithPath: $0.gazeboExecutable) }, .saved, to: &paths)
        append(
            environment["UAV_SANDBOX_GZ_EXECUTABLE"].map(URL.init(fileURLWithPath:)),
            .environment, to: &paths
        )
        paths += inspector.executableCandidates("gz", environment: environment).map {
            CandidatePath($0, .path)
        }
        return group("gazebo", "Gazebo", paths, selected) { path in
            inspector.gazebo(
                candidates: [path], manifest: manifest, root: root, profile: profile
            )
        }
    }

    private func formulaGroup(
        id: String, title: String, rule: RuntimeCompatibilityManifest.Formula,
        variable: String, manual: URL?, saved: String?, root: URL,
        selected: [String: String]
    ) -> RuntimeCandidateGroup {
        var paths: [CandidatePath] = []
        append(manual, .manual, to: &paths)
        append(saved.map(URL.init(fileURLWithPath:)), .saved, to: &paths)
        append(environment[variable].map(URL.init(fileURLWithPath:)), .environment, to: &paths)
        for base in ["/opt/homebrew/opt", "/usr/local/opt"] {
            paths.append(CandidatePath(
                URL(fileURLWithPath: base).appendingPathComponent(rule.preferredFormula),
                .homebrew
            ))
        }
        if let discovered = inspector.formula(
            id: id, title: title, rule: rule, environment: environment, root: root
        ).path {
            paths.append(CandidatePath(URL(fileURLWithPath: discovered), .homebrew))
        }
        return group(id, title, paths, selected) { path in
            guard fileManager.fileExists(atPath: path.path) else {
                return RuntimeComponent(
                    id: id, title: title, status: .missing, path: path.path,
                    detail: "Candidate path does not exist."
                )
            }
            var exact = environment
            exact[variable] = path.path
            return inspector.formula(
                id: id, title: title, rule: rule, environment: exact, root: root
            )
        }
    }

    private func group(
        _ id: String, _ title: String, _ values: [CandidatePath],
        _ selected: [String: String], inspect: (URL) -> RuntimeComponent
    ) -> RuntimeCandidateGroup {
        var seen = Set<String>()
        let candidates = values.compactMap { value -> RuntimeCandidate? in
            let path = value.url.standardizedFileURL.path
            guard seen.insert(path).inserted else { return nil }
            let item = inspect(value.url.standardizedFileURL)
            return RuntimeCandidate(
                componentID: id, title: title, origin: value.origin,
                path: path, status: item.status, version: item.version,
                detail: item.detail, selected: selected[id] == path
            )
        }
        return RuntimeCandidateGroup(id: id, title: title, candidates: candidates)
    }

    private func append(
        _ value: URL?, _ origin: RuntimeCandidateOrigin,
        to values: inout [CandidatePath]
    ) {
        if let value { values.append(CandidatePath(value, origin)) }
    }
}

private struct CandidatePath {
    let url: URL
    let origin: RuntimeCandidateOrigin

    init(_ url: URL, _ origin: RuntimeCandidateOrigin) {
        self.url = url
        self.origin = origin
    }
}
