import Foundation

public enum RuntimeCompatibilityStatus: String, Codable, Sendable {
    case compatible
    case compatibleWithWarning = "compatible_with_warning"
    case untested
    case unsupported
    case missing
    case changedSinceValidation = "changed_since_validation"

    public func blocks(_ profile: SandboxProfile) -> Bool {
        switch self {
        case .missing, .unsupported, .changedSinceValidation:
            return true
        case .untested:
            return profile == .formal
        case .compatible, .compatibleWithWarning:
            return false
        }
    }
}

public struct RuntimeComponent: Codable, Equatable, Identifiable, Sendable {
    public let id: String
    public let title: String
    public let status: RuntimeCompatibilityStatus
    public let path: String?
    public let version: String?
    public let identity: String?
    public let detail: String

    public init(
        id: String,
        title: String,
        status: RuntimeCompatibilityStatus,
        path: String? = nil,
        version: String? = nil,
        identity: String? = nil,
        detail: String
    ) {
        self.id = id
        self.title = title
        self.status = status
        self.path = path
        self.version = version
        self.identity = identity
        self.detail = detail
    }
}

public struct VerifiedRuntimeProfile: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let projectRoot: String
    public let pythonExecutable: String
    public let pythonVersion: String
    public let px4Root: String
    public let px4Commit: String
    public let gazeboExecutable: String
    public let gazeboVersion: String
    public let gazeboDistribution: String
    public let openCVPrefix: String
    public let openCVVersion: String
    public let qtPrefix: String
    public let qtVersion: String
    public let validationTimestamp: Date
    public let compatibilityResult: RuntimeCompatibilityStatus

    public init(
        schemaVersion: Int,
        projectRoot: String,
        pythonExecutable: String,
        pythonVersion: String,
        px4Root: String,
        px4Commit: String,
        gazeboExecutable: String,
        gazeboVersion: String,
        gazeboDistribution: String,
        openCVPrefix: String,
        openCVVersion: String,
        qtPrefix: String,
        qtVersion: String,
        validationTimestamp: Date,
        compatibilityResult: RuntimeCompatibilityStatus
    ) {
        self.schemaVersion = schemaVersion
        self.projectRoot = projectRoot
        self.pythonExecutable = pythonExecutable
        self.pythonVersion = pythonVersion
        self.px4Root = px4Root
        self.px4Commit = px4Commit
        self.gazeboExecutable = gazeboExecutable
        self.gazeboVersion = gazeboVersion
        self.gazeboDistribution = gazeboDistribution
        self.openCVPrefix = openCVPrefix
        self.openCVVersion = openCVVersion
        self.qtPrefix = qtPrefix
        self.qtVersion = qtVersion
        self.validationTimestamp = validationTimestamp
        self.compatibilityResult = compatibilityResult
    }
}

public struct RuntimeAssessment: Equatable, Sendable {
    public let components: [RuntimeComponent]
    public let overall: RuntimeCompatibilityStatus
    public let blockers: [String]
    public let selected: VerifiedRuntimeProfile?

    public var ready: Bool { blockers.isEmpty && selected != nil }

    public init(
        components: [RuntimeComponent],
        overall: RuntimeCompatibilityStatus,
        blockers: [String],
        selected: VerifiedRuntimeProfile?
    ) {
        self.components = components
        self.overall = overall
        self.blockers = blockers
        self.selected = selected
    }
}

public struct RuntimeSelection: Equatable, Sendable {
    public var pythonExecutable: URL?
    public var px4Root: URL?
    public var gazeboExecutable: URL?

    public init(
        pythonExecutable: URL? = nil,
        px4Root: URL? = nil,
        gazeboExecutable: URL? = nil
    ) {
        self.pythonExecutable = pythonExecutable
        self.px4Root = px4Root
        self.gazeboExecutable = gazeboExecutable
    }
}

public struct RuntimeCompatibilityManifest: Decodable, Sendable {
    public struct Python: Decodable, Sendable {
        public let minimumVersion: String
        public let testedMinorVersions: [String]
        public let requiredDevelopmentPackages: [String]
        public let optionalMlPackages: [String]
    }

    public struct PX4: Decodable, Sendable {
        public let versionPolicy: String
        public let testedCommits: [String]
        public let requiredPaths: [String]
    }

    public struct Gazebo: Decodable, Sendable {
        public let expectedDistribution: String
        public let testedSimMajorVersions: [Int]
    }

    public struct Formula: Decodable, Sendable {
        public let preferredFormula: String
        public let expectedMajorVersion: Int
    }

    public let schemaVersion: Int
    public let python: Python
    public let px4: PX4
    public let gazebo: Gazebo
    public let opencv: Formula
    public let qt: Formula

    public static func load(projectRoot: URL) throws -> Self {
        let path = projectRoot.appendingPathComponent(
            "config/sandbox/runtime_compatibility.json"
        )
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let value = try decoder.decode(Self.self, from: Data(contentsOf: path))
        guard value.schemaVersion == 1 else {
            throw RuntimeManifestError.unsupportedSchema(value.schemaVersion)
        }
        return value
    }
}

public enum RuntimeManifestError: LocalizedError, Equatable {
    case unsupportedSchema(Int)

    public var errorDescription: String? {
        switch self {
        case let .unsupportedSchema(version):
            return "Unsupported runtime compatibility manifest schema: \(version)."
        }
    }
}
