import Foundation

public protocol RuntimeCommandRunning: Sendable {
    func run(_ executable: URL, _ arguments: [String], at directory: URL) throws -> ProcessResult
}

public struct LocalRuntimeCommandRunner: RuntimeCommandRunning {
    public init() {}

    public func run(
        _ executable: URL,
        _ arguments: [String],
        at directory: URL
    ) throws -> ProcessResult {
        try ProcessExecution.run(
            executable: executable, arguments: arguments, directory: directory
        )
    }
}

public struct RuntimeInspector: @unchecked Sendable {
    private let runner: any RuntimeCommandRunning
    private let fileManager: FileManager

    public init(
        runner: any RuntimeCommandRunning = LocalRuntimeCommandRunner(),
        fileManager: FileManager = .default
    ) {
        self.runner = runner
        self.fileManager = fileManager
    }

    public func python(
        candidates: [URL], manifest: RuntimeCompatibilityManifest, root: URL,
        profile: SandboxProfile
    ) -> RuntimeComponent {
        var bestUntested: RuntimeComponent?
        var firstFailure: RuntimeComponent?
        for candidate in unique(candidates) where fileManager.isExecutableFile(atPath: candidate.path) {
            let inspected = inspectPython(candidate, manifest: manifest, root: root)
            if inspected.status == .compatible { return inspected }
            if inspected.status == .untested, bestUntested == nil { bestUntested = inspected }
            if firstFailure == nil { firstFailure = inspected }
        }
        return bestUntested ?? firstFailure ?? RuntimeComponent(
            id: "python", title: "Python", status: .missing,
            detail: "No compatible Python 3.11+ runtime with mavsdk was found."
        )
    }

    public func px4(candidates: [URL], manifest: RuntimeCompatibilityManifest) -> RuntimeComponent {
        var firstUntested: RuntimeComponent?
        var firstFailure: RuntimeComponent?
        for root in unique(candidates) where fileManager.fileExists(atPath: root.path) {
            let inspected = inspectPX4(root, manifest: manifest)
            if inspected.status == .compatible { return inspected }
            if inspected.status == .untested, firstUntested == nil { firstUntested = inspected }
            if firstFailure == nil { firstFailure = inspected }
        }
        return firstUntested ?? firstFailure ?? RuntimeComponent(
            id: "px4", title: "PX4", status: .missing,
            detail: "No valid PX4 source checkout with the required Gazebo structure was found."
        )
    }

    public func gazebo(
        candidates: [URL], manifest: RuntimeCompatibilityManifest, root: URL,
        profile: SandboxProfile
    ) -> RuntimeComponent {
        var firstUntested: RuntimeComponent?
        var firstFailure: RuntimeComponent?
        for executable in unique(candidates)
        where fileManager.isExecutableFile(atPath: executable.path) {
            let inspected = inspectGazebo(executable, manifest: manifest, root: root)
            if inspected.status == .compatible { return inspected }
            if inspected.status == .untested, firstUntested == nil { firstUntested = inspected }
            if firstFailure == nil { firstFailure = inspected }
        }
        return firstUntested ?? firstFailure ?? RuntimeComponent(
            id: "gazebo", title: "Gazebo", status: .missing,
            detail: "Gazebo Sim was not found or installed candidates are unsupported."
        )
    }

    public func formula(
        id: String, title: String, rule: RuntimeCompatibilityManifest.Formula,
        environment: [String: String], root: URL
    ) -> RuntimeComponent {
        guard let prefix = formulaPrefix(rule.preferredFormula, environment: environment, root: root)
        else {
            return RuntimeComponent(
                id: id, title: title, status: .missing,
                detail: "\(rule.preferredFormula) was not found."
            )
        }
        let version = formulaVersion(
            rule.preferredFormula, prefix: prefix,
            environment: environment, root: root
        )
        let major = version.flatMap { Int($0.split(separator: ".").first ?? "") }
        let status: RuntimeCompatibilityStatus = major == rule.expectedMajorVersion
            ? .compatible : major == nil ? .untested : .unsupported
        return RuntimeComponent(
            id: id, title: title, status: status, path: prefix.path,
            version: version, identity: "\(prefix.path)|\(version ?? "unknown")",
            detail: version.map { "\(title) \($0) at \(prefix.path)" }
                ?? "\(title) at \(prefix.path); version could not be verified."
        )
    }

    private func inspectPython(
        _ executable: URL, manifest: RuntimeCompatibilityManifest, root: URL
    ) -> RuntimeComponent {
        let packages = manifest.python.requiredDevelopmentPackages
        let script = """
        import importlib.util,sys
        print('.'.join(map(str,sys.version_info[:3])))
        print(','.join(name for name in \(String(reflecting: packages)) if importlib.util.find_spec(name) is None))
        """
        guard let result = try? runner.run(executable, ["-c", script], at: root),
              result.exitCode == 0 else {
            return RuntimeComponent(
                id: "python", title: "Python", status: .unsupported,
                path: executable.path, detail: "Python executable could not be inspected."
            )
        }
        let lines = result.output.split(whereSeparator: \.isNewline).map(String.init)
        guard let version = lines.first else {
            return RuntimeComponent(
                id: "python", title: "Python", status: .unsupported,
                path: executable.path, detail: "Python did not report a version."
            )
        }
        let missing = lines.count > 1 ? lines[1] : ""
        let minor = version.split(separator: ".").prefix(2).joined(separator: ".")
        let meetsMinimum = versionIsAtLeast(version, manifest.python.minimumVersion)
        let status: RuntimeCompatibilityStatus
        if !meetsMinimum || !missing.isEmpty {
            status = .unsupported
        } else if manifest.python.testedMinorVersions.contains(minor) {
            status = .compatible
        } else {
            status = .untested
        }
        let suffix = missing.isEmpty ? "" : "; missing packages: \(missing)"
        return RuntimeComponent(
            id: "python", title: "Python", status: status,
            path: executable.path, version: version,
            identity: "\(executable.path)|\(version)",
            detail: "Python \(version) at \(executable.path)\(suffix)"
        )
    }

    private func inspectPX4(
        _ root: URL, manifest: RuntimeCompatibilityManifest
    ) -> RuntimeComponent {
        let required = manifest.px4.requiredPaths.map { root.appendingPathComponent($0).path }
        guard required.allSatisfy({ fileManager.fileExists(atPath: $0) }),
              fileManager.fileExists(atPath: root.appendingPathComponent(".git").path)
        else {
            return RuntimeComponent(
                id: "px4", title: "PX4", status: .unsupported,
                path: root.path, detail: "PX4 checkout is missing required repository files."
            )
        }
        let git = URL(fileURLWithPath: "/usr/bin/git")
        guard let head = try? runner.run(git, ["-C", root.path, "rev-parse", "HEAD"], at: root),
              head.exitCode == 0 else {
            return RuntimeComponent(
                id: "px4", title: "PX4", status: .unsupported,
                path: root.path, detail: "PX4 Git identity could not be read."
            )
        }
        let commit = head.output.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = (try? runner.run(
            git, ["-C", root.path, "describe", "--tags", "--always"], at: root
        ))?.output.trimmingCharacters(in: .whitespacesAndNewlines)
        let status: RuntimeCompatibilityStatus = manifest.px4.testedCommits.contains(commit)
            ? .compatible : .untested
        return RuntimeComponent(
            id: "px4", title: "PX4", status: status, path: root.path,
            version: ref?.isEmpty == false ? ref : nil, identity: commit,
            detail: "PX4 \(String(commit.prefix(12))) at \(root.path)"
        )
    }

    private func inspectGazebo(
        _ executable: URL, manifest: RuntimeCompatibilityManifest, root: URL
    ) -> RuntimeComponent {
        guard let result = try? runner.run(executable, ["sim", "--versions"], at: root),
              result.exitCode == 0 else {
            return RuntimeComponent(
                id: "gazebo", title: "Gazebo", status: .unsupported,
                path: executable.path, detail: "gz exists but Gazebo Sim inspection failed."
            )
        }
        let version = firstVersion(in: result.output)
        let major = version.flatMap { Int($0.split(separator: ".").first ?? "") }
        let status: RuntimeCompatibilityStatus = major.map {
            manifest.gazebo.testedSimMajorVersions.contains($0) ? .compatible : .unsupported
        } ?? .untested
        return RuntimeComponent(
            id: "gazebo", title: "Gazebo", status: status,
            path: executable.path, version: version,
            identity: "\(executable.path)|\(version ?? "unknown")",
            detail: version.map {
                "\(manifest.gazebo.expectedDistribution) / Gazebo Sim \($0) at \(executable.path)"
            } ?? "Gazebo at \(executable.path); distribution could not be verified."
        )
    }

    private func formulaPrefix(
        _ name: String, environment: [String: String], root: URL
    ) -> URL? {
        let variable = name.hasPrefix("opencv")
            ? "UAV_SANDBOX_OPENCV_PREFIX" : "UAV_SANDBOX_QT_PREFIX"
        if let explicit = environment[variable], !explicit.isEmpty {
            let candidate = URL(fileURLWithPath: explicit).standardizedFileURL
            if fileManager.fileExists(atPath: candidate.path) { return candidate }
        }
        for base in ["/opt/homebrew/opt", "/usr/local/opt"] {
            let candidate = URL(fileURLWithPath: base).appendingPathComponent(name)
            if fileManager.fileExists(atPath: candidate.path) { return candidate }
        }
        for brew in brewCandidates(environment) where fileManager.isExecutableFile(atPath: brew.path) {
            if let result = try? runner.run(brew, ["--prefix", name], at: root) {
                let value = result.output.trimmingCharacters(in: .whitespacesAndNewlines)
                if !value.isEmpty { return URL(fileURLWithPath: value).standardizedFileURL }
            }
        }
        return nil
    }

    private func formulaVersion(
        _ name: String, prefix: URL,
        environment: [String: String], root: URL
    ) -> String? {
        for brew in brewCandidates(environment) where fileManager.isExecutableFile(atPath: brew.path) {
            if let result = try? runner.run(brew, ["list", "--versions", name], at: root) {
                for line in result.output.split(whereSeparator: \.isNewline) {
                    let fields = line.split(whereSeparator: \.isWhitespace)
                    if fields.first == Substring(name), fields.count > 1,
                       firstVersion(in: String(fields[1])) != nil {
                        return String(fields[1])
                    }
                }
            }
        }
        let resolved = prefix.resolvingSymlinksInPath().lastPathComponent
        return firstVersion(in: resolved)
    }

    private func brewCandidates(_ environment: [String: String]) -> [URL] {
        executableCandidates("brew", environment: environment)
    }

    public func executableCandidates(
        _ name: String, environment: [String: String]
    ) -> [URL] {
        let search = environment["PATH"] ?? "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
        return unique(search.split(separator: ":").map {
            URL(fileURLWithPath: String($0)).appendingPathComponent(name)
        } + [URL(fileURLWithPath: "/opt/homebrew/bin/\(name)"),
             URL(fileURLWithPath: "/usr/local/bin/\(name)")])
    }

    private func unique(_ values: [URL]) -> [URL] {
        var seen = Set<String>()
        return values.map(\.standardizedFileURL).filter { seen.insert($0.path).inserted }
    }

    private func versionIsAtLeast(_ value: String, _ minimum: String) -> Bool {
        let actual = value.split(separator: ".").prefix(3).map { Int($0) ?? 0 }
        let required = minimum.split(separator: ".").prefix(3).map { Int($0) ?? 0 }
        for index in 0..<max(actual.count, required.count) {
            let left = index < actual.count ? actual[index] : 0
            let right = index < required.count ? required[index] : 0
            if left != right { return left > right }
        }
        return true
    }

    private func firstVersion(in value: String) -> String? {
        value.split(whereSeparator: \.isWhitespace).map(String.init).first {
            $0.range(of: #"^\d+\.\d+(\.\d+)?$"#, options: .regularExpression) != nil
        }
    }
}
