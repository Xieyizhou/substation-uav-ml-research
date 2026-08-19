import Foundation

enum WorkbenchImageInference {
    static func isExperimentID(_ value: String) -> Bool {
        value.range(
            of: "^[a-z0-9][a-z0-9-]{0,63}$", options: .regularExpression
        ) != nil
    }

    static func stage(source: URL, projectRoot: URL) throws -> String {
        let values = try source.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard values.isRegularFile == true, (values.fileSize ?? 0) <= 50 * 1024 * 1024 else {
            throw NSError(domain: "UAVSandbox", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Choose a regular PNG/JPEG no larger than 50 MiB."
            ])
        }
        let suffix = source.pathExtension.lowercased()
        guard ["png", "jpg", "jpeg"].contains(suffix) else {
            throw NSError(domain: "UAVSandbox", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Only PNG and JPEG images are supported."
            ])
        }
        let inbox = projectRoot.appendingPathComponent(
            "outputs/sandbox/workbench/inbox", isDirectory: true
        )
        try FileManager.default.createDirectory(at: inbox, withIntermediateDirectories: true)
        let name = "image-\(UUID().uuidString.lowercased()).\(suffix)"
        try FileManager.default.copyItem(at: source, to: inbox.appendingPathComponent(name))
        return name
    }

    static func start(
        webURL: URL, stagedName: String, experimentID: String,
        comparisonExperimentID: String?
    ) async throws {
        let operatorURL = webURL.appendingPathComponent("api/operator")
        let (data, response) = try await URLSession.shared.data(from: operatorURL)
        guard (response as? HTTPURLResponse)?.statusCode == 200,
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let token = object["operator_token"] as? String else {
            throw AppFailure.startupTimeout
        }
        var parameters: [String: Any] = [
            "staged_name": stagedName, "experiment_id": experimentID,
        ]
        if let comparisonExperimentID, !comparisonExperimentID.isEmpty {
            parameters["comparison_experiment_id"] = comparisonExperimentID
        }
        var request = URLRequest(url: operatorURL.appendingPathComponent("start"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(token, forHTTPHeaderField: "X-Sandbox-Token")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "action": "workbench-image-infer", "parameters": parameters,
        ])
        let (result, postResponse) = try await URLSession.shared.data(for: request)
        guard (postResponse as? HTTPURLResponse)?.statusCode == 202 else {
            let message = (try? JSONSerialization.jsonObject(with: result))
                .flatMap { $0 as? [String: Any] }?["error"] as? String
            throw NSError(domain: "UAVSandbox", code: 4, userInfo: [
                NSLocalizedDescriptionKey: message ?? "Image inference was rejected."
            ])
        }
    }
}
