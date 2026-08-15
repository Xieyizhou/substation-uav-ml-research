import CryptoKit
import Foundation

public enum StandaloneDemo {
    private struct Sample {
        let distance: Double
        let density: Double
        let label: String
    }

    private struct Payload: Codable {
        let schemaVersion: Int
        let algorithm: String
        let scopeNote: String
        let trainingSampleCount: Int
        let evaluationSampleCount: Int
        let accuracy: Double
        let macroF1: Double
        let passed: Bool
        let predictions: [DemoPrediction]

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "demo_result_schema_version"
            case algorithm, predictions, accuracy, passed
            case scopeNote = "scope_note"
            case trainingSampleCount = "training_sample_count"
            case evaluationSampleCount = "evaluation_sample_count"
            case macroF1 = "macro_f1"
        }
    }

    private static let training = [
        Sample(distance: 3.8, density: 0.08, label: "safe"),
        Sample(distance: 3.1, density: 0.12, label: "safe"),
        Sample(distance: 2.7, density: 0.18, label: "safe"),
        Sample(distance: 2.3, density: 0.22, label: "safe"),
        Sample(distance: 1.2, density: 0.64, label: "danger"),
        Sample(distance: 0.9, density: 0.73, label: "danger"),
        Sample(distance: 0.7, density: 0.82, label: "danger"),
        Sample(distance: 0.5, density: 0.91, label: "danger"),
    ]

    private static let evaluation = [
        Sample(distance: 3.5, density: 0.10, label: "safe"),
        Sample(distance: 2.9, density: 0.16, label: "safe"),
        Sample(distance: 2.5, density: 0.20, label: "safe"),
        Sample(distance: 2.0, density: 0.28, label: "safe"),
        Sample(distance: 1.4, density: 0.58, label: "danger"),
        Sample(distance: 1.0, density: 0.70, label: "danger"),
        Sample(distance: 0.8, density: 0.79, label: "danger"),
        Sample(distance: 0.4, density: 0.94, label: "danger"),
    ]

    public static func run(outputDirectory: URL? = nil) throws -> StandaloneDemoResult {
        let centers = centroids()
        let predictions = evaluation.enumerated().map { index, sample in
            DemoPrediction(
                sampleID: String(format: "demo-%02d", index),
                minimumDistanceM: sample.distance,
                obstacleDensity: sample.density,
                expected: sample.label,
                predicted: predict(sample, centers: centers)
            )
        }
        let scores = metrics(predictions)
        let payload = Payload(
            schemaVersion: 1,
            algorithm: "nearest_centroid_v1",
            scopeNote: "Illustrative synthetic data; not formal research evidence.",
            trainingSampleCount: training.count,
            evaluationSampleCount: evaluation.count,
            accuracy: scores.accuracy,
            macroF1: scores.macroF1,
            passed: predictions.allSatisfy { $0.expected == $0.predicted },
            predictions: predictions
        )
        let identity = sha256(try encoder.encode(payload))
        let result = StandaloneDemoResult(
            schemaVersion: payload.schemaVersion,
            algorithm: payload.algorithm,
            scopeNote: payload.scopeNote,
            trainingSampleCount: payload.trainingSampleCount,
            evaluationSampleCount: payload.evaluationSampleCount,
            accuracy: payload.accuracy,
            macroF1: payload.macroF1,
            passed: payload.passed,
            predictions: payload.predictions,
            artifactIdentitySHA256: identity
        )
        if let outputDirectory { try persist(result, to: outputDirectory) }
        return result
    }

    private static var encoder: JSONEncoder {
        let value = JSONEncoder()
        value.outputFormatting = [.prettyPrinted, .sortedKeys]
        return value
    }

    private static func persist(_ result: StandaloneDemoResult, to directory: URL) throws {
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        try encoder.encode(result).write(
            to: directory.appendingPathComponent("demo_result.json"),
            options: .atomic
        )
    }

    private static func centroids() -> [String: (Double, Double)] {
        Dictionary(uniqueKeysWithValues: ["safe", "danger"].map { label in
            let rows = training.filter { $0.label == label }
            return (label, (
                rows.map(\.distance).reduce(0, +) / Double(rows.count),
                rows.map(\.density).reduce(0, +) / Double(rows.count)
            ))
        })
    }

    private static func predict(
        _ sample: Sample,
        centers: [String: (Double, Double)]
    ) -> String {
        centers.min { left, right in
            let leftDistance = squaredDistance(sample, center: left.value)
            let rightDistance = squaredDistance(sample, center: right.value)
            return leftDistance == rightDistance
                ? left.key < right.key
                : leftDistance < rightDistance
        }!.key
    }

    private static func squaredDistance(
        _ sample: Sample,
        center: (Double, Double)
    ) -> Double {
        pow(sample.distance - center.0, 2) + pow(sample.density - center.1, 2)
    }

    private static func metrics(
        _ predictions: [DemoPrediction]
    ) -> (accuracy: Double, macroF1: Double) {
        let labels = ["safe", "danger"]
        let correct = predictions.filter { $0.expected == $0.predicted }.count
        let f1 = labels.map { label -> Double in
            let truePositive = predictions.filter {
                $0.expected == label && $0.predicted == label
            }.count
            let falsePositive = predictions.filter {
                $0.expected != label && $0.predicted == label
            }.count
            let falseNegative = predictions.filter {
                $0.expected == label && $0.predicted != label
            }.count
            let denominator = 2 * truePositive + falsePositive + falseNegative
            return denominator == 0 ? 0 : Double(2 * truePositive) / Double(denominator)
        }
        return (
            Double(correct) / Double(predictions.count),
            f1.reduce(0, +) / Double(f1.count)
        )
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
