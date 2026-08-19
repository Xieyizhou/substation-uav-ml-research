import Foundation
import SandboxAppCore
import XCTest

final class RuntimeCandidateTests: XCTestCase {
func testDiscoveryListsAllPythonCandidatesAndMarksTheSelectedRuntime() throws {
    let fixture = try RuntimeFixture()
    let venv = try fixture.executable(".venv/bin/python")
    let old = try fixture.executable("path/python3")
    fixture.runner.pythonVersions[venv.path] = "3.14.1"
    fixture.runner.pythonVersions[old.path] = "3.9.18"
    let manager = fixture.manager(path: old.deletingLastPathComponent())
    let assessment = try manager.assess(
        projectRoot: fixture.root, profile: .development,
        selection: fixture.selection, enforceSavedIdentity: false
    )
    let groups = try manager.discoverCandidates(
        projectRoot: fixture.root, profile: .development,
        selection: fixture.selection, assessment: assessment
    )
    let candidates = try require(groups.first { $0.id == "python" }?.candidates)
    expect(candidates.contains { $0.path == venv.path && $0.selected })
    expect(candidates.contains { $0.path == old.path && $0.status == .unsupported })
}

func testManualFormulaPrefixCanBeSelectedAndPersisted() throws {
    let fixture = try RuntimeFixture()
    let python = try fixture.executable(".venv/bin/python")
    fixture.runner.pythonVersions[python.path] = "3.14.1"
    let manual = fixture.root.appendingPathComponent("custom-opencv/4.14.0")
    try FileManager.default.createDirectory(at: manual, withIntermediateDirectories: true)
    let selection = RuntimeSelection(
        px4Root: fixture.px4, gazeboExecutable: fixture.gz,
        openCVPrefix: manual
    )
    let manager = fixture.manager(path: fixture.root.appendingPathComponent("empty"))
    let assessment = try manager.validateAndPersist(
        projectRoot: fixture.root, profile: .development, selection: selection
    )
    expect(assessment.selected?.openCVPrefix == manual.path)
    expect(fixture.store.load()?.openCVPrefix == manual.path)
}

func testUnrelatedFormulaDirectoryCannotBorrowHomebrewVersion() throws {
    let fixture = try RuntimeFixture()
    let python = try fixture.executable(".venv/bin/python")
    fixture.runner.pythonVersions[python.path] = "3.14.1"
    let unrelated = fixture.root.appendingPathComponent("unrelated-opencv")
    try FileManager.default.createDirectory(at: unrelated, withIntermediateDirectories: true)
    let selection = RuntimeSelection(
        px4Root: fixture.px4, gazeboExecutable: fixture.gz,
        openCVPrefix: unrelated
    )
    let assessment = try fixture.manager(path: fixture.root.appendingPathComponent("empty"))
        .assess(
            projectRoot: fixture.root, profile: .development,
            selection: selection, enforceSavedIdentity: false
        )
    let opencv = assessment.components.first { $0.id == "opencv" }
    expect(opencv?.path == unrelated.path)
    expect(opencv?.status == .untested)
    expect(assessment.selected == nil)
}

func testMissingManualFormulaPrefixBlocksInsteadOfFallingBack() throws {
    let fixture = try RuntimeFixture()
    let python = try fixture.executable(".venv/bin/python")
    fixture.runner.pythonVersions[python.path] = "3.14.1"
    let missing = fixture.root.appendingPathComponent("missing-qt")
    let selection = RuntimeSelection(
        px4Root: fixture.px4, gazeboExecutable: fixture.gz,
        qtPrefix: missing
    )
    let assessment = try fixture.manager(path: fixture.root.appendingPathComponent("empty"))
        .assess(
            projectRoot: fixture.root, profile: .development,
            selection: selection, enforceSavedIdentity: false
        )
    let qt = assessment.components.first { $0.id == "qt" }
    expect(qt?.status == .missing)
    expect(!assessment.ready)
}

func testDuplicateCandidatePathsAreCollapsedWithStablePriority() throws {
    let fixture = try RuntimeFixture()
    let python = try fixture.executable(".venv/bin/python")
    fixture.runner.pythonVersions[python.path] = "3.14.1"
    let manager = fixture.manager(path: fixture.gz.deletingLastPathComponent())
    let assessment = try manager.assess(
        projectRoot: fixture.root, profile: .development,
        selection: fixture.selection, enforceSavedIdentity: false
    )
    let groups = try manager.discoverCandidates(
        projectRoot: fixture.root, profile: .development,
        selection: fixture.selection, assessment: assessment
    )
    let gazebo = try require(groups.first { $0.id == "gazebo" }?.candidates)
    expect(gazebo.filter { $0.path == fixture.gz.path }.count == 1)
    expect(gazebo.first { $0.path == fixture.gz.path }?.origin == .manual)
}
}
