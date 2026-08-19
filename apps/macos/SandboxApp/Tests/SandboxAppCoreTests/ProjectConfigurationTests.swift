import Foundation
import SandboxAppCore
import XCTest

final class ProjectConfigurationTests: XCTestCase {
    func testProjectUsesVerifiedAbsoluteRuntimePaths() throws {
        let root = try temporaryProject()
        let runtime = testRuntime(root: root)
        let project = try ProjectLocator.locate(root: root, runtime: runtime)
        expect(project.root == root.standardizedFileURL)
        expect(project.python.path == runtime.pythonExecutable)
        expect(project.arguments(
            profile: .development, port: 8765, command: "serve"
        ) == [
            root.appendingPathComponent("main.py").path,
            "sandbox", "--project-root", root.path, "--profile", "development",
            "serve", "--host", "127.0.0.1", "--port", "8765",
        ])
    }

    func testRejectsFolderWithoutProjectEntryPoint() {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        expectThrows(ProjectValidationError.missingMainScript) {
            try ProjectLocator.validate(root: root)
        }
    }

    func testEnvironmentProvidesSuggestedRoot() {
        let root = URL(fileURLWithPath: "/tmp/example-sandbox")
        expect(ProjectLocator.suggestedRoot(
            environment: ["UAV_SANDBOX_PROJECT_ROOT": root.path]
        ) == root)
    }

    func testRuntimeEnvironmentPutsVerifiedToolsFirst() throws {
        let root = try temporaryProject()
        let runtime = testRuntime(root: root)
        let project = try ProjectLocator.locate(root: root, runtime: runtime)
        let environment = project.runtimeEnvironment(base: [
            "PATH": "/usr/bin:/bin", "EXISTING": "preserved",
        ])
        let folders = environment["PATH"]?.split(separator: ":").map(String.init)
        expect(Array(folders?.prefix(4) ?? []) == [
            URL(fileURLWithPath: runtime.pythonExecutable).deletingLastPathComponent().path,
            URL(fileURLWithPath: runtime.gazeboExecutable).deletingLastPathComponent().path,
            runtime.qtPrefix + "/bin", runtime.openCVPrefix + "/bin",
        ])
        expect(environment["PX4_ROOT"] == runtime.px4Root)
        expect(environment["UAV_SANDBOX_GZ_EXECUTABLE"] == runtime.gazeboExecutable)
        expect(environment["UAV_SANDBOX_PYTHON"] == runtime.pythonExecutable)
        expect(environment["EXISTING"] == "preserved")
    }
}

func temporaryProject() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(
        at: root.appendingPathComponent("src/sandbox"), withIntermediateDirectories: true
    )
    FileManager.default.createFile(
        atPath: root.appendingPathComponent("main.py").path, contents: Data()
    )
    return root.standardizedFileURL
}

func testRuntime(root: URL) -> VerifiedRuntimeProfile {
    VerifiedRuntimeProfile(
        schemaVersion: 1, projectRoot: root.path,
        pythonExecutable: root.appendingPathComponent(".venv/bin/python").path,
        pythonVersion: "3.13.2", px4Root: "/tmp/PX4-Autopilot",
        px4Commit: String(repeating: "a", count: 40),
        gazeboExecutable: "/opt/gazebo/bin/gz", gazeboVersion: "8.9.0",
        gazeboDistribution: "Harmonic", openCVPrefix: "/opt/opencv@4",
        openCVVersion: "4.10.0", qtPrefix: "/opt/qt@5", qtVersion: "5.15.16",
        validationTimestamp: Date(timeIntervalSince1970: 0),
        compatibilityResult: .compatible
    )
}
