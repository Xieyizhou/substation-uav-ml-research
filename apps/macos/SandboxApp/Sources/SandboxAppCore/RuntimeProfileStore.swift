import Foundation

public struct RuntimeProfileStore: Sendable {
    public let fileURL: URL

    public init(fileURL: URL? = nil, fileManager: FileManager = .default) {
        if let fileURL {
            self.fileURL = fileURL
            return
        }
        let support = fileManager.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first ?? fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support")
        self.fileURL = support
            .appendingPathComponent("UAV Research Sandbox", isDirectory: true)
            .appendingPathComponent("runtime-profile.json")
    }

    public func load(fileManager: FileManager = .default) -> VerifiedRuntimeProfile? {
        guard fileManager.isReadableFile(atPath: fileURL.path) else { return nil }
        do {
            let data = try Data(contentsOf: fileURL)
            guard let value = decode(data, snakeCase: true)
                    ?? decode(data, snakeCase: false) else { return nil }
            return value.schemaVersion == 1 ? value : nil
        } catch {
            return nil
        }
    }

    public func save(
        _ profile: VerifiedRuntimeProfile,
        fileManager: FileManager = .default
    ) throws {
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        encoder.keyEncodingStrategy = .custom { path in
            let key = path.last?.stringValue ?? ""
            return ProfileCodingKey(Self.snakeKeys[key] ?? key)
        }
        let data = try encoder.encode(profile)
        try data.write(to: fileURL, options: .atomic)
    }

    private func decode(
        _ data: Data, snakeCase: Bool
    ) -> VerifiedRuntimeProfile? {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        if snakeCase {
            decoder.keyDecodingStrategy = .custom { path in
                let key = path.last?.stringValue ?? ""
                return ProfileCodingKey(Self.camelKeys[key] ?? key)
            }
        }
        return try? decoder.decode(VerifiedRuntimeProfile.self, from: data)
    }

    private static let snakeKeys = [
        "schemaVersion": "schema_version",
        "projectRoot": "project_root",
        "pythonExecutable": "python_executable",
        "pythonVersion": "python_version",
        "px4Root": "px4_root",
        "px4Commit": "px4_commit",
        "gazeboExecutable": "gazebo_executable",
        "gazeboVersion": "gazebo_version",
        "gazeboDistribution": "gazebo_distribution",
        "openCVPrefix": "opencv_prefix",
        "openCVVersion": "opencv_version",
        "qtPrefix": "qt_prefix",
        "qtVersion": "qt_version",
        "validationTimestamp": "validation_timestamp",
        "compatibilityResult": "compatibility_result",
    ]

    private static let camelKeys = Dictionary(
        uniqueKeysWithValues: snakeKeys.map { ($0.value, $0.key) }
    )
}

private struct ProfileCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int? = nil

    init(_ stringValue: String) { self.stringValue = stringValue }
    init?(stringValue: String) { self.init(stringValue) }
    init?(intValue: Int) { return nil }
}
