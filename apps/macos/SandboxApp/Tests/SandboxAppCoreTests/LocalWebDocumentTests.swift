import Foundation
import SandboxAppCore
import Testing

@Suite struct LocalWebDocumentTests {
    @Test func rootTrailingSlashDoesNotTriggerReload() {
        let target = URL(string: "http://127.0.0.1:8765")!
        let loaded = URL(string: "http://127.0.0.1:8765/")!

        #expect(LocalWebDocument.isSame(current: loaded, target: target))
    }

    @Test func tabFragmentDoesNotTriggerReload() {
        let target = URL(string: "http://127.0.0.1:8765")!
        let workbench = URL(string: "http://127.0.0.1:8765/#experiments")!

        #expect(LocalWebDocument.isSame(current: workbench, target: target))
    }

    @Test func differentDocumentStillLoads() {
        let target = URL(string: "http://127.0.0.1:8765")!

        #expect(!LocalWebDocument.isSame(
            current: URL(string: "http://127.0.0.1:8766/"), target: target
        ))
        #expect(!LocalWebDocument.isSame(
            current: URL(string: "http://127.0.0.1:8765/setup"), target: target
        ))
    }
}
