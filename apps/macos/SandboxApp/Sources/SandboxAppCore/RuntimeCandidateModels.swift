import Foundation

public enum RuntimeCandidateOrigin: String, Codable, Sendable {
    case repository
    case manual
    case saved
    case environment
    case path = "PATH"
    case defaultLocation = "default"
    case homebrew = "Homebrew"
}

public struct RuntimeCandidate: Equatable, Identifiable, Sendable {
    public let componentID: String
    public let title: String
    public let origin: RuntimeCandidateOrigin
    public let path: String
    public let status: RuntimeCompatibilityStatus
    public let version: String?
    public let detail: String
    public let selected: Bool

    public var id: String { "\(componentID)|\(path)" }

    public init(
        componentID: String,
        title: String,
        origin: RuntimeCandidateOrigin,
        path: String,
        status: RuntimeCompatibilityStatus,
        version: String?,
        detail: String,
        selected: Bool
    ) {
        self.componentID = componentID
        self.title = title
        self.origin = origin
        self.path = path
        self.status = status
        self.version = version
        self.detail = detail
        self.selected = selected
    }
}

public struct RuntimeCandidateGroup: Equatable, Identifiable, Sendable {
    public let id: String
    public let title: String
    public let candidates: [RuntimeCandidate]

    public init(id: String, title: String, candidates: [RuntimeCandidate]) {
        self.id = id
        self.title = title
        self.candidates = candidates
    }
}
