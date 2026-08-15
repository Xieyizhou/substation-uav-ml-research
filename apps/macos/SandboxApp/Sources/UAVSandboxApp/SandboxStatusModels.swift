import Foundation

struct SandboxStatusSnapshot {
    let version: VersionSnapshot
    let setup: SetupSnapshot
    let profile: ProfileSnapshot
    let runtime: [RuntimeSnapshot]
    let checks: [DoctorSnapshot]
    let storage: StorageSnapshot
    let operatorStatus: OperatorSnapshot

    var aliveProcessCount: Int { runtime.filter(\.alive).count }
    var passedCheckCount: Int { checks.filter { $0.status == "pass" }.count }
    var warningCheckCount: Int { checks.filter { $0.status == "warning" }.count }
    var failedCheckCount: Int { checks.filter { $0.status == "failure" }.count }
}

struct SetupSnapshot: Decodable {
    let ready: Bool
    let requiredReadyCount: Int
    let requiredCount: Int

    enum CodingKeys: String, CodingKey {
        case ready
        case requiredReadyCount = "required_ready_count"
        case requiredCount = "required_count"
    }
}

struct VersionSnapshot: Decodable {
    let sandboxProductVersion: String
    let macosAppVersion: String
    let operatorAPIVersion: String
    let gateSchemaVersion: Int

    enum CodingKeys: String, CodingKey {
        case sandboxProductVersion = "sandbox_product_version"
        case macosAppVersion = "macos_app_version"
        case operatorAPIVersion = "operator_api_version"
        case gateSchemaVersion = "gate_schema_version"
    }
}

struct ProfileSnapshot: Decodable {
    let profileID: String
    let title: String
    let description: String
    let availableWorkflows: [String]
    let flightEnabled: Bool
    let formalEvidence: Bool

    enum CodingKeys: String, CodingKey {
        case profileID = "profile_id"
        case title, description
        case availableWorkflows = "available_workflows"
        case flightEnabled = "flight_enabled"
        case formalEvidence = "formal_evidence"
    }
}

struct RuntimeSnapshot: Decodable, Identifiable {
    let name: String
    let available: Bool
    let alive: Bool
    let pid: Int?
    let detail: String

    var id: String { name }
}

struct DoctorSnapshot: Decodable, Identifiable {
    let name: String
    let status: String
    let explanation: String
    let action: String

    var id: String { name }
}

struct StorageSnapshot: Decodable {
    let profile: String
    let sandboxOutputRoot: String
    let sandboxOutputBytes: Int64
    let diskFreeBytes: Int64
    let diskTotalBytes: Int64
    let minimumFreeBytes: Int64
    let formalRetentionLocked: Bool

    enum CodingKeys: String, CodingKey {
        case profile
        case sandboxOutputRoot = "sandbox_output_root"
        case sandboxOutputBytes = "sandbox_output_bytes"
        case diskFreeBytes = "disk_free_bytes"
        case diskTotalBytes = "disk_total_bytes"
        case minimumFreeBytes = "minimum_free_bytes"
        case formalRetentionLocked = "formal_retention_locked"
    }
}

struct OperatorSnapshot: Decodable {
    let state: String
    let activeJob: JobSnapshot?
    let history: [JobSnapshot]

    enum CodingKeys: String, CodingKey {
        case state
        case activeJob = "active_job"
        case history
    }
}

struct JobSnapshot: Decodable, Identifiable {
    let jobID: String
    let action: String
    let state: String
    let createdAt: String
    let scenarioID: String?
    let error: String?
    let failureCode: String?

    var id: String { jobID }

    enum CodingKeys: String, CodingKey {
        case jobID = "job_id"
        case action, state, error
        case createdAt = "created_at"
        case scenarioID = "scenario_id"
        case failureCode = "failure_code"
    }
}
