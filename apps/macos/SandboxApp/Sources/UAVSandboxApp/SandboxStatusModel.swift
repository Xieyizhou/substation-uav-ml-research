import Foundation

@MainActor
final class SandboxStatusModel: ObservableObject {
    enum Phase: Equatable {
        case idle
        case loading
        case ready
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var snapshot: SandboxStatusSnapshot?
    @Published private(set) var lastUpdated: Date?

    private var refreshTask: Task<Void, Never>?

    deinit { refreshTask?.cancel() }

    func start(baseURL: URL) {
        refreshTask?.cancel()
        refreshTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.refresh(baseURL: baseURL)
                try? await Task.sleep(nanoseconds: 5_000_000_000)
            }
        }
    }

    func stop() {
        refreshTask?.cancel()
        refreshTask = nil
        phase = .idle
        snapshot = nil
        lastUpdated = nil
    }

    func refresh(baseURL: URL) async {
        if snapshot == nil { phase = .loading }
        do {
            let profile: ProfileSnapshot = try await fetch("api/profile", from: baseURL)
            let version: VersionSnapshot = try await fetch("api/version", from: baseURL)
            let runtime: [RuntimeSnapshot] = try await fetch("api/runtime", from: baseURL)
            let checks: [DoctorSnapshot] = try await fetch("api/doctor", from: baseURL)
            let storage: StorageSnapshot = try await fetch("api/storage", from: baseURL)
            let operatorStatus: OperatorSnapshot = try await fetch("api/operator", from: baseURL)
            snapshot = SandboxStatusSnapshot(
                version: version,
                profile: profile,
                runtime: runtime,
                checks: checks,
                storage: storage,
                operatorStatus: operatorStatus
            )
            lastUpdated = Date()
            phase = .ready
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private func fetch<Value: Decodable>(_ path: String, from baseURL: URL) async throws -> Value {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.timeoutInterval = 3
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            throw StatusFailure.unavailable(path)
        }
        return try JSONDecoder().decode(Value.self, from: data)
    }
}

private enum StatusFailure: LocalizedError {
    case unavailable(String)

    var errorDescription: String? {
        switch self {
        case let .unavailable(path): return "Status endpoint \(path) is unavailable."
        }
    }
}
