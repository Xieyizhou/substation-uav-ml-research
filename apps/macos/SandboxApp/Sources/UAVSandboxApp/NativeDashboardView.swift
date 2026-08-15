import SwiftUI

struct NativeDashboardView: View {
    @ObservedObject var status: SandboxStatusModel
    let refresh: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                title
                switch status.phase {
                case .idle where status.snapshot == nil,
                     .loading where status.snapshot == nil:
                    loading
                case let .failed(message) where status.snapshot == nil:
                    failure(message)
                default:
                    if let snapshot = status.snapshot { dashboard(snapshot) }
                }
            }
            .padding(28)
            .frame(maxWidth: 1180, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var title: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 5) {
                Text("Sandbox status")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                Text("A read-only view of the local service, environment, and managed job.")
                    .foregroundStyle(.secondary)
                if let version = status.snapshot?.version {
                    Text("App \(version.macosAppVersion) · Sandbox \(version.sandboxProductVersion)")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            if let updated = status.lastUpdated {
                Text(updated, style: .time)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Button(action: refresh) {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
        }
    }

    private var loading: some View {
        HStack(spacing: 12) {
            ProgressView()
            Text("Reading local Sandbox status…")
        }
        .frame(maxWidth: .infinity, minHeight: 260)
    }

    private func failure(_ message: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 42, weight: .light))
                .foregroundStyle(.orange)
            Text("Status unavailable").font(.title2.weight(.semibold))
            Text(message).foregroundStyle(.secondary)
            Button("Try again", action: refresh)
        }
        .frame(maxWidth: .infinity, minHeight: 300)
    }

    private func dashboard(_ value: SandboxStatusSnapshot) -> some View {
        VStack(spacing: 18) {
            HStack(spacing: 14) {
                MetricCard(
                    title: value.profile.title,
                    value: value.setup.ready ? "Environment ready" : "Setup needed",
                    detail: "\(value.setup.requiredReadyCount) / \(value.setup.requiredCount) requirements",
                    symbol: "square.stack.3d.up",
                    tint: value.setup.ready ? .green : .orange
                )
                MetricCard(
                    title: "Managed operator",
                    value: value.operatorStatus.state.capitalized,
                    detail: value.operatorStatus.activeJob?.action ?? "No active job",
                    symbol: "switch.2",
                    tint: value.operatorStatus.activeJob == nil ? .green : .orange
                )
                MetricCard(
                    title: "Runtime",
                    value: "\(value.aliveProcessCount) / \(value.runtime.count)",
                    detail: "Detected processes",
                    symbol: "waveform.path.ecg",
                    tint: value.aliveProcessCount > 0 ? .green : .secondary
                )
                MetricCard(
                    title: "Free storage",
                    value: ByteCountFormatter.string(
                        fromByteCount: value.storage.diskFreeBytes,
                        countStyle: .file
                    ),
                    detail: "Sandbox uses \(ByteCountFormatter.string(fromByteCount: value.storage.sandboxOutputBytes, countStyle: .file))",
                    symbol: "internaldrive",
                    tint: value.storage.diskFreeBytes >= value.storage.minimumFreeBytes ? .green : .red
                )
            }
            HStack(alignment: .top, spacing: 18) {
                statusCard("Environment", symbol: "checkmark.shield") {
                    checkSummary(value)
                    Divider()
                    ForEach(value.checks) { item in DoctorRow(item: item) }
                }
                statusCard("Runtime processes", symbol: "terminal") {
                    ForEach(value.runtime) { item in RuntimeRow(item: item) }
                }
            }
            if let job = value.operatorStatus.activeJob ?? value.operatorStatus.history.first {
                jobCard(job, active: value.operatorStatus.activeJob != nil)
            }
        }
    }

    private func checkSummary(_ value: SandboxStatusSnapshot) -> some View {
        HStack(spacing: 12) {
            Label("\(value.passedCheckCount) passed", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
            Label("\(value.warningCheckCount) warnings", systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            if value.failedCheckCount > 0 {
                Label("\(value.failedCheckCount) failed", systemImage: "xmark.octagon.fill")
                    .foregroundStyle(.red)
            }
            Spacer()
        }
        .font(.caption.weight(.semibold))
    }

    private func statusCard<Content: View>(
        _ title: String,
        symbol: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: symbol).font(.headline)
            content()
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background, in: RoundedRectangle(cornerRadius: 14))
        .overlay { RoundedRectangle(cornerRadius: 14).stroke(.separator) }
    }

    private func jobCard(_ job: JobSnapshot, active: Bool) -> some View {
        statusCard(active ? "Active job" : "Most recent job", symbol: "clock.arrow.circlepath") {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(job.action).font(.headline)
                    Text(job.scenarioID ?? job.jobID)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text(job.state.capitalized)
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 10).padding(.vertical, 5)
                    .background(.quaternary, in: Capsule())
            }
            if let error = job.error {
                Label(error, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
            }
        }
    }
}

private struct MetricCard: View {
    let title: String
    let value: String
    let detail: String
    let symbol: String
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: symbol)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value).font(.title3.weight(.bold)).lineLimit(1)
            Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(1)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(tint.opacity(0.09), in: RoundedRectangle(cornerRadius: 14))
        .overlay { RoundedRectangle(cornerRadius: 14).stroke(tint.opacity(0.25)) }
    }
}

private struct DoctorRow: View {
    let item: DoctorSnapshot
    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: item.status == "pass" ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                .foregroundStyle(item.status == "pass" ? .green : item.status == "warning" ? .orange : .red)
            VStack(alignment: .leading, spacing: 2) {
                Text(item.name).font(.subheadline.weight(.semibold))
                Text(item.explanation).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }
}

private struct RuntimeRow: View {
    let item: RuntimeSnapshot
    var body: some View {
        HStack(spacing: 10) {
            Circle().fill(item.alive ? .green : .secondary).frame(width: 8, height: 8)
            Text(item.name).font(.subheadline.weight(.semibold))
            Spacer()
            Text(item.pid.map { "PID \($0)" } ?? item.detail)
                .font(.caption).foregroundStyle(.secondary).lineLimit(1)
        }
        .padding(.vertical, 3)
    }
}
