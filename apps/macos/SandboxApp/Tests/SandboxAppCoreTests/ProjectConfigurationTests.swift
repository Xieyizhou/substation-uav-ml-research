import Foundation
import SandboxAppCore
import Testing

@Test func locatesVirtualEnvironmentPython() throws {
    let root = try temporaryProject(withPython: true)
    let project = try ProjectLocator.locate(root: root, environment: ["PATH": ""])
    #expect(project.root == root.standardizedFileURL)
    #expect(project.python.path.hasSuffix(".venv/bin/python"))
    #expect(project.arguments(profile: .demo, port: 8765, command: "serve") == [
        root.appendingPathComponent("main.py").path,
        "sandbox", "--project-root", root.path, "--profile", "demo", "serve",
        "--host", "127.0.0.1", "--port", "8765",
    ])
}

@Test func rejectsFolderWithoutProjectEntryPoint() {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
    #expect(throws: ProjectValidationError.missingMainScript) {
        try ProjectLocator.locate(root: root, environment: ["PATH": ""])
    }
}

@Test func environmentProvidesSuggestedRoot() {
    let root = URL(fileURLWithPath: "/tmp/example-sandbox")
    #expect(ProjectLocator.suggestedRoot(
        environment: ["UAV_SANDBOX_PROJECT_ROOT": root.path]
    ) == root)
}

private func temporaryProject(withPython: Bool) throws -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(
        at: root.appendingPathComponent("src/sandbox"),
        withIntermediateDirectories: true
    )
    FileManager.default.createFile(
        atPath: root.appendingPathComponent("main.py").path,
        contents: Data()
    )
    if withPython {
        let python = root.appendingPathComponent(".venv/bin/python")
        try FileManager.default.createDirectory(
            at: python.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        FileManager.default.createFile(atPath: python.path, contents: Data("#!\n".utf8))
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], atPath: python.path
        )
    }
    return root
}
