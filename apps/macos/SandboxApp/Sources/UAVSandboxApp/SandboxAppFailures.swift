import Foundation

struct ProfileResponse: Decodable {
    let profileID: String

    enum CodingKeys: String, CodingKey {
        case profileID = "profile_id"
    }
}

enum AppFailure: LocalizedError {
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
