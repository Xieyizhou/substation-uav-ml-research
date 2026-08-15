import AppKit
import Combine
import Darwin
import Foundation
import SandboxAppCore

@MainActor
final class SandboxAppModel: ObservableObject {
    enum State: Equatable {
        case idle
        case preparing
        case starting
        case online
        case connected
        case stopping
        case failed(String)

        var label: String {
            switch self {
            case .idle: return "Idle"
            case .preparing: return "Preparing"
            case .starting: return "Starting"
            case .online: return "Online"
            case .connected: return "Connected"
            case .stopping: return "Stopping"
            case .failed: return "Needs attention"
            }
        }
    }

    @Published var state: State = .idle
    @Published var profile: SandboxProfile = .demo
    @Published var projectRoot = ""
    @Published var logLines: [String] = []
    @Published var webURL: URL?

    private let port: UInt16 = 8765
    private var process: Process?
    private var outputPipe: Pipe?
    private var ownsServer = false

    init() {
        let stored = UserDefaults.standard.string(forKey: "sandboxProjectRoot")
        let workingRoot = ProjectLocator.suggestedRoot()
        let bundleRoot = Bundle.main.executableURL.flatMap {
            ProjectLocator.suggestedRoot(currentDirectory: $0)
        }
        projectRoot = stored ?? workingRoot?.path ?? bundleRoot?.path ?? ""
    }

    var canStart: Bool {
        switch state {
        case .idle, .failed: return !projectRoot.isEmpty
        default: return false
        }
    }

    var canStop: Bool {
        switch state {
        case .online, .connected: return true
        default: return false
        }
    }

    var statusDetail: String {
        if case let .failed(message) = state { return message }
        switch state {
        case .online: return "Local service is managed by this App."
        case .connected: return "Using an existing local service. Stop will only disconnect."
        case .preparing: return "Validating the project and initializing the profile."
        case .starting: return "Waiting for the loopback service to become ready."
        case .stopping: return "Requesting graceful shutdown."
        default: return "Select the repository and start a profile."
        }
    }

    func chooseProject() {
        let panel = NSOpenPanel()
        panel.title = "Choose the substation UAV project"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            projectRoot = url.path
            UserDefaults.standard.set(url.path, forKey: "sandboxProjectRoot")
            if case .failed = state { state = .idle }
        }
    }

    func start() {
        guard canStart else { return }
        state = .preparing
        logLines = []
        webURL = nil
        let root = URL(fileURLWithPath: projectRoot)
        let selectedProfile = profile
        let selectedPort = port
        Task {
            do {
                let project = try ProjectLocator.locate(root: root)
                UserDefaults.standard.set(project.root.path, forKey: "sandboxProjectRoot")
                if let runningProfile = await runningProfile() {
                    guard runningProfile == selectedProfile.rawValue else {
                        throw AppFailure.profileConflict(
                            running: runningProfile,
                            selected: selectedProfile.rawValue
                        )
                    }
                    ownsServer = false
                    webURL = serverURL
                    state = .connected
                    append("Connected to the existing loopback Sandbox service.")
                    return
                }
                let result = try await Task.detached {
                    try ProcessExecution.run(
                        executable: project.python,
                        arguments: project.arguments(
                            profile: selectedProfile,
                            port: selectedPort,
                            command: "bootstrap"
                        ),
                        directory: project.root,
                        environment: project.runtimeEnvironment()
                    )
                }.value
                guard result.exitCode == 0 else {
                    throw AppFailure.bootstrap(result.output)
                }
                append(result.output)
                try launch(project: project, profile: selectedProfile)
                state = .starting
                guard await waitUntilReady(expectedProfile: selectedProfile) else {
                    stopOwnedProcess()
                    throw AppFailure.startupTimeout
                }
                webURL = serverURL
                state = .online
                append("Sandbox service is ready at \(serverURL.absoluteString)")
            } catch {
                state = .failed(error.localizedDescription)
                append("Start failed: \(error.localizedDescription)")
            }
        }
    }

    func stop() {
        guard canStop else { return }
        if !ownsServer {
            webURL = nil
            state = .idle
            append("Disconnected without stopping the external service.")
            return
        }
        state = .stopping
        let ownedProcess = process
        Task {
            await Task.detached { Self.stopGracefully(ownedProcess) }.value
            finishStoppedProcess()
        }
    }

    func shutdownBeforeApplicationExit() {
        guard ownsServer else { return }
        Self.stopGracefully(process)
        finishStoppedProcess()
    }

    private var serverURL: URL {
        URL(string: "http://127.0.0.1:\(port)")!
    }

    private func launch(project: SandboxProject, profile: SandboxProfile) throws {
        let child = Process()
        let pipe = Pipe()
        child.executableURL = project.python
        child.arguments = project.arguments(profile: profile, port: port, command: "serve")
        child.currentDirectoryURL = project.root
        child.environment = project.runtimeEnvironment()
        child.standardOutput = pipe
        child.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let text = String(decoding: data, as: UTF8.self)
            Task { @MainActor in self?.append(text) }
        }
        child.terminationHandler = { [weak self] process in
            Task { @MainActor in
                self?.serverExited(code: process.terminationStatus)
            }
        }
        try child.run()
        process = child
        outputPipe = pipe
        ownsServer = true
        append("Started Sandbox service process \(child.processIdentifier).")
    }

    private func endpointReady(expectedProfile: SandboxProfile) async -> Bool {
        await runningProfile() == expectedProfile.rawValue
    }

    private func runningProfile() async -> String? {
        var request = URLRequest(url: serverURL.appendingPathComponent("api/profile"))
        request.timeoutInterval = 1
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                return nil
            }
            let profile = try JSONDecoder().decode(ProfileResponse.self, from: data)
            return profile.profileID
        } catch {
            return nil
        }
    }

    private func waitUntilReady(expectedProfile: SandboxProfile) async -> Bool {
        for _ in 0..<100 {
            if await endpointReady(expectedProfile: expectedProfile) { return true }
            if process?.isRunning != true { return false }
            try? await Task.sleep(nanoseconds: 150_000_000)
        }
        return false
    }

    private func append(_ text: String) {
        let incoming = text.split(whereSeparator: \.isNewline).map(String.init)
        guard !incoming.isEmpty else { return }
        logLines.append(contentsOf: incoming)
        if logLines.count > 500 { logLines.removeFirst(logLines.count - 500) }
    }

    private func serverExited(code: Int32) {
        guard ownsServer else { return }
        let wasStopping = state == .stopping
        finishStoppedProcess()
        if !wasStopping && code != 0 {
            state = .failed("Sandbox service exited with code \(code).")
        }
    }

    private func stopOwnedProcess() {
        Self.stopGracefully(process)
        finishStoppedProcess()
    }

    private func finishStoppedProcess() {
        outputPipe?.fileHandleForReading.readabilityHandler = nil
        outputPipe = nil
        process = nil
        ownsServer = false
        webURL = nil
        if state == .stopping { state = .idle }
    }

    nonisolated private static func stopGracefully(_ process: Process?) {
        guard let process, process.isRunning else { return }
        kill(process.processIdentifier, SIGINT)
        for _ in 0..<50 where process.isRunning { usleep(100_000) }
        if process.isRunning { process.terminate() }
        for _ in 0..<20 where process.isRunning { usleep(100_000) }
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
    }
}

private struct ProfileResponse: Decodable {
    let profileID: String

    enum CodingKeys: String, CodingKey {
        case profileID = "profile_id"
    }
}

private enum AppFailure: LocalizedError {
    case bootstrap(String)
    case profileConflict(running: String, selected: String)
    case startupTimeout

    var errorDescription: String? {
        switch self {
        case let .bootstrap(output):
            return "Profile bootstrap failed. \(output.trimmingCharacters(in: .whitespacesAndNewlines))"
        case let .profileConflict(running, selected):
            return "Port 8765 already hosts the \(running) profile. Stop it or select \(running) instead of \(selected)."
        case .startupTimeout:
            return "The local Sandbox service did not become ready within 15 seconds."
        }
    }
}
