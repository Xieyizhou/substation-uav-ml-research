import Foundation

public struct DemoPrediction: Codable, Equatable, Identifiable, Sendable {
    public let sampleID: String
    public let minimumDistanceM: Double
    public let obstacleDensity: Double
    public let expected: String
    public let predicted: String

    public var id: String { sampleID }

    enum CodingKeys: String, CodingKey {
        case sampleID = "sample_id"
        case minimumDistanceM = "minimum_distance_m"
        case obstacleDensity = "obstacle_density"
        case expected, predicted
    }
}

public struct StandaloneDemoResult: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let algorithm: String
    public let scopeNote: String
    public let trainingSampleCount: Int
    public let evaluationSampleCount: Int
    public let accuracy: Double
    public let macroF1: Double
    public let passed: Bool
    public let predictions: [DemoPrediction]
    public let artifactIdentitySHA256: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "demo_result_schema_version"
        case algorithm, predictions, accuracy, passed
        case scopeNote = "scope_note"
        case trainingSampleCount = "training_sample_count"
        case evaluationSampleCount = "evaluation_sample_count"
        case macroF1 = "macro_f1"
        case artifactIdentitySHA256 = "artifact_identity_sha256"
    }
}
