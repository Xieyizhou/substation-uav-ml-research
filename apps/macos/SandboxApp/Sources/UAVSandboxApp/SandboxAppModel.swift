import AppKit
import Combine
import Darwin
import Foundation
import SandboxAppCore

@MainActor
final class SandboxAppModel: ObservableObject {
    @Published var state: State = .idle
    @Published var profile: SandboxProfile = .demo
    @Published var projectRoot = ""
    @Published var logLines: [String] = []
    @Published var webURL: URL?
    @Published var embeddedDemoReady = false
    @Published var runtimeAssessment: RuntimeAssessment?
    @Published var runtimeCandidates: [RuntimeCandidateGroup] = []
    @Published var runtimeRefreshing = false
    private let port: UInt16 = 8765
    private var process: Process?
    private var outputPipe: Pipe?
    private var ownsServer = false
    let runtimeManager = RuntimeCompatibilityManager()
    var runtimeSelection = RuntimeSelection()
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
        case .idle, .failed:
            return profile == .demo || runtimeAssessment?.ready == true
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
        case .online where embeddedDemoReady:
            return "Built-in Demo is running entirely in this App."
        case .online: return "Local service is managed by this App."
        case .connected: return "Using an existing local service. Stop will only disconnect."
        case .preparing: return "Validating the project and initializing the profile."
        case .starting: return "Waiting for the loopback service to become ready."
        case .stopping: return "Requesting graceful shutdown."
        default: return profile == .demo
            ? "Start immediately with the built-in Demo."
            : "Select the repository and start a profile."
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
            runtimeSelection = RuntimeSelection()
            runtimeAssessment = nil
            runtimeCandidates = []
            if case .failed = state { state = .idle }
            Task { await refreshRuntime() }
        }
    }

    func importYOLODataset(canonicalToSourceID: [String: Int]) {
        guard profile == .development, webURL != nil else {
            append("YOLO import requires the running Development profile.")
            return
        }
        guard canonicalToSourceID.count == 4,
              Set(canonicalToSourceID.values).count == 4 else {
            append("YOLO import requires four distinct source class IDs.")
            return
        }
        let panel = NSOpenPanel()
        panel.title = "Choose a YOLO Detect dataset"
        panel.message = "Select the folder containing dataset.yaml."
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let source = panel.url else { return }
        let datasetID = Self.importDatasetID(source.lastPathComponent)
        Task {
            do {
                try await startDatasetImport(
                    source: source, datasetID: datasetID,
                    canonicalToSourceID: canonicalToSourceID
                )
                append("Started managed YOLO dataset import as \(datasetID).")
            } catch {
                append("Dataset import failed: \(error.localizedDescription)")
            }
        }
    }

    private func startDatasetImport(
        source: URL, datasetID: String, canonicalToSourceID: [String: Int]
    ) async throws {
        guard let webURL else { throw AppFailure.startupTimeout }
        let operatorURL = webURL.appendingPathComponent("api/operator")
        let (data, response) = try await URLSession.shared.data(from: operatorURL)
        guard (response as? HTTPURLResponse)?.statusCode == 200,
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let token = object["operator_token"] as? String else {
            throw AppFailure.startupTimeout
        }
        var request = URLRequest(url: operatorURL.appendingPathComponent("start"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(token, forHTTPHeaderField: "X-Sandbox-Token")
        let sourceToCanonical = Dictionary(uniqueKeysWithValues:
            canonicalToSourceID.map { (String($0.value), $0.key) })
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "action": "workbench-dataset-import",
            "parameters": [
                "source": source.path, "dataset_id": datasetID,
                "class_map": sourceToCanonical,
            ],
        ])
        let (result, postResponse) = try await URLSession.shared.data(for: request)
        guard (postResponse as? HTTPURLResponse)?.statusCode == 202 else {
            let message = (try? JSONSerialization.jsonObject(with: result))
                .flatMap { $0 as? [String: Any] }?["error"] as? String
            throw NSError(domain: "UAVSandbox", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: message ?? "Import was rejected."])
        }
    }

    nonisolated private static func importDatasetID(_ name: String) -> String {
        let slug = name.lowercased().map { character in
            character.isLetter || character.isNumber ? character : "-"
        }
        let collapsed = String(slug).split(separator: "-").joined(separator: "-")
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return String("import-\(collapsed)-\(formatter.string(from: Date()))".prefix(64))
    }

    func start() {
        guard canStart else { return }
        prepareForStart()
        if profile == .demo {
            startEmbeddedDemo()
            return
        }
        let root = URL(fileURLWithPath: projectRoot)
        let selectedProfile = profile
        let selectedPort = port
        Task {
            do {
                let runtime = try await runtimeForLaunch(profile: selectedProfile)
                try await connectOrLaunch(
                    root: root,
                    profile: selectedProfile,
                    port: selectedPort,
                    runtime: runtime
                )
            } catch {
                state = .failed(error.localizedDescription)
                append("Start failed: \(error.localizedDescription)")
            }
        }
    }

    private func prepareForStart() {
        state = .preparing
        logLines = []
        webURL = nil
        embeddedDemoReady = false
    }

    private func startEmbeddedDemo() {
        embeddedDemoReady = true
        state = .online
        append("Started the repository-free built-in Demo runtime.")
    }

    private func connectOrLaunch(
        root: URL,
        profile: SandboxProfile,
        port: UInt16,
        runtime: VerifiedRuntimeProfile
    ) async throws {
        let project = try ProjectLocator.locate(root: root, runtime: runtime)
        UserDefaults.standard.set(project.root.path, forKey: "sandboxProjectRoot")
        if let runningProfile = await runningProfile() {
            guard runningProfile == profile.rawValue else {
                throw AppFailure.profileConflict(
                    running: runningProfile,
                    selected: profile.rawValue
                )
            }
            ownsServer = false
            webURL = serverURL
            state = .connected
            append("Connected to the existing loopback Sandbox service.")
            return
        }
        let conflicts = RuntimeConflictChecker().conflicts()
        if !conflicts.isEmpty { throw AppFailure.externalRuntimeConflict(conflicts) }
        let result = try await bootstrap(project: project, profile: profile, port: port)
        guard result.exitCode == 0 else { throw AppFailure.bootstrap(result.output) }
        append(result.output)
        try launch(project: project, profile: profile)
        state = .starting
        guard await waitUntilReady(expectedProfile: profile) else {
            stopOwnedProcess()
            throw AppFailure.startupTimeout
        }
        webURL = serverURL
        state = .online
        append("Sandbox service is ready at \(serverURL.absoluteString)")
    }

    private func bootstrap(
        project: SandboxProject,
        profile: SandboxProfile,
        port: UInt16
    ) async throws -> ProcessResult {
        try await Task.detached {
            try ProcessExecution.run(
                executable: project.python,
                arguments: project.arguments(
                    profile: profile,
                    port: port,
                    command: "bootstrap"
                ),
                directory: project.root,
                environment: project.runtimeEnvironment()
            )
        }.value
    }

    func stop() {
        guard canStop else { return }
        if embeddedDemoReady {
            embeddedDemoReady = false
            state = .idle
            append("Stopped the built-in Demo runtime.")
            return
        }
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

    func managedJobIsActive() -> Bool {
        guard ownsServer,
              let data = try? Data(contentsOf: serverURL.appendingPathComponent("api/operator")),
              let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        return value["active_job"] is [String: Any]
    }

    func detachServiceForActiveJob() {
        outputPipe?.fileHandleForReading.readabilityHandler = nil
        process?.terminationHandler = nil
        outputPipe = nil
        process = nil
        ownsServer = false
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

    func append(_ text: String) {
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
        embeddedDemoReady = false
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
