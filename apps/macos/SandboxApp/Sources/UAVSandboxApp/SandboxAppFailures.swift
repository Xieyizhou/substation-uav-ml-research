import Foundation
import SandboxAppCore

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
    case runtimeCompatibility([String])
    case externalRuntimeConflict([RuntimeConflict])

    var errorDescription: String? {
        switch self {
        case let .bootstrap(output):
            return "Profile bootstrap failed. \(output.trimmingCharacters(in: .whitespacesAndNewlines))"
        case let .profileConflict(running, selected):
            return "Port 8765 already hosts the \(running) profile. Stop it or select \(running) instead of \(selected)."
        case .startupTimeout:
            return "The local Sandbox service did not become ready within 15 seconds."
        case let .runtimeCompatibility(blockers):
            return "Runtime compatibility is blocked. \(blockers.joined(separator: " "))"
        case let .externalRuntimeConflict(conflicts):
            let values = conflicts.map { "\($0.process) PID \($0.pid)" }.joined(separator: ", ")
            return "External simulator runtime conflict detected: \(values). Stop it explicitly before launch; the App will not terminate it."
        }
    }
}
