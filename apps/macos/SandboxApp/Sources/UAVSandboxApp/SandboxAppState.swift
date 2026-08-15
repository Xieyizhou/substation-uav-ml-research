extension SandboxAppModel {
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
}
