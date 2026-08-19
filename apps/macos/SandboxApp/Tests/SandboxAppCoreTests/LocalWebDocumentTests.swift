import Foundation
import SandboxAppCore
import XCTest

final class LocalWebDocumentTests: XCTestCase {
    func testRootTrailingSlashDoesNotTriggerReload() {
        let target = URL(string: "http://127.0.0.1:8765")!
        let loaded = URL(string: "http://127.0.0.1:8765/")!

        expect(LocalWebDocument.isSame(current: loaded, target: target))
    }

    func testTabFragmentDoesNotTriggerReload() {
        let target = URL(string: "http://127.0.0.1:8765")!
        let workbench = URL(string: "http://127.0.0.1:8765/#experiments")!

        expect(LocalWebDocument.isSame(current: workbench, target: target))
    }

    func testDifferentDocumentStillLoads() {
        let target = URL(string: "http://127.0.0.1:8765")!

        expect(!LocalWebDocument.isSame(
            current: URL(string: "http://127.0.0.1:8766/"), target: target
        ))
        expect(!LocalWebDocument.isSame(
            current: URL(string: "http://127.0.0.1:8765/setup"), target: target
        ))
    }
}
