import Foundation
import SandboxAppCore
import Testing

@Suite struct StandaloneDemoTests {
    @Test func demoProfileDoesNotRequireProject() {
        #expect(SandboxProfile.demo.requiresProject == false)
        #expect(SandboxProfile.development.requiresProject)
        #expect(SandboxProfile.formal.requiresProject)
    }

    @Test func standaloneDemoIsDeterministicAndPersistsArtifact() throws {
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        let first = try StandaloneDemo.run(outputDirectory: output)
        let second = try StandaloneDemo.run()

        #expect(first == second)
        #expect(first.passed)
        #expect(first.accuracy == 1.0)
        #expect(first.macroF1 == 1.0)
        #expect(first.predictions.count == 8)
        #expect(first.artifactIdentitySHA256.count == 64)
        #expect(FileManager.default.fileExists(
            atPath: output.appendingPathComponent("demo_result.json").path
        ))
    }
}
