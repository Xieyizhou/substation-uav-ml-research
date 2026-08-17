import AppKit

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var model: SandboxAppModel?

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let model, model.managedJobIsActive() else {
            model?.shutdownBeforeApplicationExit()
            return .terminateNow
        }
        let alert = NSAlert()
        alert.messageText = "A Workbench task is still running"
        alert.informativeText = "Keep it running for recovery when the App reopens, or stop it safely and preserve its checkpoint."
        alert.addButton(withTitle: "Keep Task Running")
        alert.addButton(withTitle: "Stop Task and Quit")
        alert.addButton(withTitle: "Cancel")
        switch alert.runModal() {
        case .alertFirstButtonReturn:
            model.detachServiceForActiveJob()
            return .terminateNow
        case .alertSecondButtonReturn:
            model.shutdownBeforeApplicationExit()
            return .terminateNow
        default:
            return .terminateCancel
        }
    }
}
