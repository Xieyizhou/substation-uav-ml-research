import Foundation
import SandboxAppCore
import Testing

final class FakeRunner: RuntimeCommandRunning, @unchecked Sendable {
    var pythonVersions: [String: String] = [:]
    var px4Commits: [String: String] = [:]
    var gazeboVersions: [String: String] = [:]
    var formulaPrefixes: [String: String] = [:]
    var formulaVersions: [String: String] = [:]
    var processList = ""

    func run(_ executable: URL, _ arguments: [String], at directory: URL) throws -> ProcessResult {
        let path = executable.path
        if arguments.first == "-c", let version = pythonVersions[path] {
            return ProcessResult(exitCode: 0, output: "\(version)\n\n")
        }
        if executable.lastPathComponent == "git", arguments.contains("rev-parse") {
            let root = arguments[arguments.firstIndex(of: "-C")! + 1]
            return ProcessResult(exitCode: 0, output: "\(px4Commits[root] ?? "unknown")\n")
        }
        if executable.lastPathComponent == "git", arguments.contains("describe") {
            return ProcessResult(exitCode: 0, output: "v1.test\n")
        }
        if arguments == ["sim", "--versions"] {
            let value = gazeboVersions[path]
            return ProcessResult(exitCode: value == nil ? 1 : 0, output: value ?? "")
        }
        if arguments.first == "--prefix", let name = arguments.last,
           let prefix = formulaPrefixes[name] {
            return ProcessResult(exitCode: 0, output: prefix + "\n")
        }
        if arguments.prefix(2) == ["list", "--versions"], let name = arguments.last,
           let version = formulaVersions[name] {
            return ProcessResult(exitCode: 0, output: "\(name) \(version)\n")
        }
        if executable.path == "/bin/ps" {
            return ProcessResult(exitCode: 0, output: processList)
        }
        return ProcessResult(exitCode: 1, output: "")
    }
}

@Suite struct RuntimeCompatibilityTests {
@Test func repositoryVenvWinsOverPathPython() throws {
    let fixture = try RuntimeFixture()
    let fallback = try fixture.executable("bin/python3")
    let venv = try fixture.executable(".venv/bin/python")
    fixture.runner.pythonVersions[fallback.path] = "3.13.1"
    fixture.runner.pythonVersions[venv.path] = "3.14.1"
    let result = try fixture.manager(path: fallback.deletingLastPathComponent())
        .validateAndPersist(projectRoot: fixture.root, profile: .development,
                            selection: fixture.selection)
    #expect(result.selected?.pythonExecutable == venv.path)
}

@Test func incompatibleVenvDoesNotHideCompatiblePython() throws {
    let fixture = try RuntimeFixture()
    let fallback = try fixture.executable("bin/python3")
    let venv = try fixture.executable(".venv/bin/python")
    fixture.runner.pythonVersions[venv.path] = "3.9.18"
    fixture.runner.pythonVersions[fallback.path] = "3.13.4"
    let result = try fixture.manager(path: fallback.deletingLastPathComponent())
        .validateAndPersist(projectRoot: fixture.root, profile: .development,
                            selection: fixture.selection)
    #expect(result.selected?.pythonExecutable == fallback.path)
}

@Test func missingSavedPythonRequiresExplicitRevalidation() throws {
    let fixture = try RuntimeFixture()
    let fallback = try fixture.executable("bin/python3")
    fixture.runner.pythonVersions[fallback.path] = "3.13.4"
    try fixture.store.save(fixture.saved(python: fixture.root.appendingPathComponent("gone")))
    let result = try fixture.manager(path: fallback.deletingLastPathComponent())
        .assess(projectRoot: fixture.root, profile: .development,
                selection: fixture.selection)
    #expect(result.overall == .changedSinceValidation)
    #expect(!result.ready)
}

@Test func savedPythonWinsOverRandomPathFallback() throws {
    let fixture = try RuntimeFixture()
    let savedPython = try fixture.executable("saved/bin/python")
    let fallback = try fixture.executable("path/bin/python3")
    fixture.runner.pythonVersions[savedPython.path] = "3.13.4"
    fixture.runner.pythonVersions[fallback.path] = "3.14.1"
    try fixture.store.save(fixture.saved(python: savedPython))
    let result = try fixture.manager(path: fallback.deletingLastPathComponent())
        .assess(projectRoot: fixture.root, profile: .development,
                selection: fixture.selection)
    #expect(result.selected?.pythonExecutable == savedPython.path)
}

@Test func gazeboSkipsUnsupportedFirstPathCandidate() throws {
    let fixture = try RuntimeFixture()
    let wrong = try fixture.executable("wrong/gz")
    let right = try fixture.executable("right/gz")
    fixture.runner.gazeboVersions[wrong.path] = "7.9.0\n"
    fixture.runner.gazeboVersions[right.path] = "8.14.0\n"
    let manifest = try RuntimeCompatibilityManifest.load(projectRoot: fixture.root)
    let item = RuntimeInspector(runner: fixture.runner).gazebo(
        candidates: [wrong, right], manifest: manifest,
        root: fixture.root, profile: .development
    )
    #expect(item.path == right.path)
    #expect(item.status == .compatible)
}

@Test func verifiedGazeboCandidateWinsOverPathFallback() throws {
    let fixture = try RuntimeFixture()
    let selected = try fixture.executable("selected/gz")
    let fallback = try fixture.executable("fallback/gz")
    fixture.runner.gazeboVersions[selected.path] = "8.13.0\n"
    fixture.runner.gazeboVersions[fallback.path] = "8.14.0\n"
    let manifest = try RuntimeCompatibilityManifest.load(projectRoot: fixture.root)
    let item = RuntimeInspector(runner: fixture.runner).gazebo(
        candidates: [selected, fallback], manifest: manifest,
        root: fixture.root, profile: .development
    )
    #expect(item.path == selected.path)
}

@Test func invalidPX4DirectoryIsUnsupported() throws {
    let fixture = try RuntimeFixture()
    let invalid = fixture.root.appendingPathComponent("invalid-px4")
    try FileManager.default.createDirectory(at: invalid, withIntermediateDirectories: true)
    let manifest = try RuntimeCompatibilityManifest.load(projectRoot: fixture.root)
    let item = RuntimeInspector(runner: fixture.runner).px4(
        candidates: [invalid], manifest: manifest
    )
    #expect(item.status == .unsupported)
}

@Test func px4HeadChangeIsDetected() throws {
    let fixture = try RuntimeFixture()
    let python = try fixture.executable(".venv/bin/python")
    fixture.runner.pythonVersions[python.path] = "3.14.1"
    let manager = fixture.manager(path: fixture.root.appendingPathComponent("empty"))
    _ = try manager.validateAndPersist(
        projectRoot: fixture.root, profile: .development, selection: fixture.selection
    )
    fixture.runner.px4Commits[fixture.px4.path] = String(repeating: "b", count: 40)
    let changed = try manager.assess(
        projectRoot: fixture.root, profile: .development, selection: fixture.selection
    )
    #expect(changed.overall == .changedSinceValidation)
}

@Test func profileStoreIgnoresCorruption() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let store = RuntimeProfileStore(fileURL: root.appendingPathComponent("profile.json"))
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try Data("not-json".utf8).write(to: store.fileURL)
    #expect(store.load() == nil)
}

@Test func profileStoreRoundTripsAndRejectsUnknownSchema() throws {
    let fixture = try RuntimeFixture()
    let profile = fixture.saved(python: try fixture.executable(".venv/bin/python"))
    try fixture.store.save(profile)
    #expect(fixture.store.load() == profile)

    let unsupported = try String(
        contentsOf: fixture.store.fileURL, encoding: .utf8
    ).replacingOccurrences(of: "\"schema_version\" : 1", with: "\"schema_version\" : 99")
    try Data(unsupported.utf8).write(to: fixture.store.fileURL)
    #expect(fixture.store.load() == nil)
}

@Test func conflictCheckerOnlyReportsSimulatorProcesses() {
    let runner = FakeRunner()
    runner.processList = """
      101 /usr/bin/zsh zsh -c echo px4
      202 /tmp/px4 /tmp/px4
      303 /opt/homebrew/bin/gz /opt/homebrew/bin/gz sim -r world.sdf
      404 /opt/homebrew/bin/gz /opt/homebrew/bin/gz topic -l
    """
    let conflicts = RuntimeConflictChecker(runner: runner).conflicts()
    #expect(conflicts.map(\.pid) == [202, 303])
}

@Test func profilePoliciesKeepDemoIndependentAndFormalStrict() {
    #expect(!RuntimeCompatibilityStatus.untested.blocks(.development))
    #expect(RuntimeCompatibilityStatus.untested.blocks(.formal))
}

@Test func formalReportsUntestedPythonInsteadOfMissing() throws {
    let fixture = try RuntimeFixture()
    let python = try fixture.executable(".venv/bin/python")
    fixture.runner.pythonVersions[python.path] = "3.12.9"
    let result = try fixture.manager(path: fixture.root.appendingPathComponent("empty"))
        .assess(projectRoot: fixture.root, profile: .formal,
                selection: fixture.selection, enforceSavedIdentity: false)
    let component = result.components.first { $0.id == "python" }
    #expect(component?.status == .untested)
    #expect(!result.ready)
}

@Test func manifestRejectsUnknownSchema() throws {
    let fixture = try RuntimeFixture()
    let path = fixture.root.appendingPathComponent("config/sandbox/runtime_compatibility.json")
    let unsupported = try String(contentsOf: path, encoding: .utf8)
        .replacingOccurrences(of: "\"schema_version\":1", with: "\"schema_version\":99")
    try Data(unsupported.utf8).write(to: path)
    #expect(throws: RuntimeManifestError.unsupportedSchema(99)) {
        try RuntimeCompatibilityManifest.load(projectRoot: fixture.root)
    }
}
}

final class RuntimeFixture {
    let root: URL
    let runner = FakeRunner()
    let px4: URL
    let gz: URL
    let opencv: URL
    let qt: URL
    let commit = "6e569d87b977d0579966a4cae78abc275aa9aaf3"
    let store: RuntimeProfileStore

    init() throws {
        root = try temporaryProject()
        store = RuntimeProfileStore(fileURL: root.appendingPathComponent("profile.json"))
        px4 = root.appendingPathComponent("PX4-Autopilot")
        gz = root.appendingPathComponent("gazebo/bin/gz")
        opencv = root.appendingPathComponent("opencv@4")
        qt = root.appendingPathComponent("qt@5")
        try writeManifest()
        for path in ["Makefile", "CMakeLists.txt", "Tools/simulation/gz/.keep", ".git/HEAD"] {
            let file = px4.appendingPathComponent(path)
            try FileManager.default.createDirectory(
                at: file.deletingLastPathComponent(), withIntermediateDirectories: true
            )
            FileManager.default.createFile(atPath: file.path, contents: Data())
        }
        _ = try executable("gazebo/bin/gz")
        _ = try executable("brew/bin/brew")
        runner.px4Commits[px4.path] = commit
        runner.gazeboVersions[gz.path] = "8.14.0\n"
        runner.formulaPrefixes = ["opencv@4": opencv.path, "qt@5": qt.path]
        runner.formulaVersions = ["opencv@4": "4.14.0", "qt@5": "5.15.19"]
    }

    var selection: RuntimeSelection {
        RuntimeSelection(px4Root: px4, gazeboExecutable: gz)
    }

    func manager(path: URL) -> RuntimeCompatibilityManager {
        let brew = root.appendingPathComponent("brew/bin")
        return RuntimeCompatibilityManager(
            store: store, inspector: RuntimeInspector(runner: runner),
            environment: [
                "PATH": "\(path.path):\(brew.path)",
                "UAV_SANDBOX_OPENCV_PREFIX": opencv.path,
                "UAV_SANDBOX_QT_PREFIX": qt.path,
            ]
        )
    }

    func executable(_ relative: String) throws -> URL {
        let file = root.appendingPathComponent(relative)
        try FileManager.default.createDirectory(
            at: file.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        FileManager.default.createFile(atPath: file.path, contents: Data("#!/bin/sh\n".utf8))
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: file.path
        )
        return file
    }

    func saved(python: URL) -> VerifiedRuntimeProfile {
        VerifiedRuntimeProfile(
            schemaVersion: 1, projectRoot: root.path,
            pythonExecutable: python.path, pythonVersion: "3.13.4",
            px4Root: px4.path, px4Commit: commit,
            gazeboExecutable: gz.path, gazeboVersion: "8.14.0",
            gazeboDistribution: "Harmonic", openCVPrefix: opencv.path,
            openCVVersion: "4.14.0", qtPrefix: qt.path, qtVersion: "5.15.19",
            validationTimestamp: Date(timeIntervalSince1970: 0),
            compatibilityResult: .compatible
        )
    }

    private func writeManifest() throws {
        let destination = root.appendingPathComponent("config/sandbox/runtime_compatibility.json")
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        let value = """
        {"schema_version":1,
         "python":{"minimum_version":"3.11","tested_minor_versions":["3.11","3.13","3.14"],"required_development_packages":["mavsdk"],"optional_ml_packages":[]},
         "px4":{"version_policy":"tested_commits_or_warning","tested_commits":["\(commit)"],"required_paths":["Makefile","CMakeLists.txt","Tools/simulation/gz"]},
         "gazebo":{"expected_distribution":"Harmonic","tested_sim_major_versions":[8]},
         "opencv":{"preferred_formula":"opencv@4","expected_major_version":4},
         "qt":{"preferred_formula":"qt@5","expected_major_version":5}}
        """
        try Data(value.utf8).write(to: destination)
    }
}
