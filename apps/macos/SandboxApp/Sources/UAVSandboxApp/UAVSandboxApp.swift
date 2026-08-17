import SwiftUI

@main
struct UAVSandboxApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var model = SandboxAppModel()

    var body: some Scene {
        WindowGroup("UAV Research Sandbox") {
            ContentView(model: model)
                .frame(minWidth: 980, minHeight: 680)
                .onAppear { appDelegate.model = model }
        }
        .windowStyle(.titleBar)
        .commands {
            CommandGroup(after: .appInfo) {
                Button("Start Sandbox") { model.start() }
                    .disabled(!model.canStart)
                Button("Stop Sandbox") { model.stop() }
                    .disabled(!model.canStop)
            }
        }
    }
}
