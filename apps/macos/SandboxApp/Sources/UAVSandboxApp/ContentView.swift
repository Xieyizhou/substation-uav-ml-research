import AppKit
import SandboxAppCore
import SwiftUI

struct ContentView: View {
    @ObservedObject var model: SandboxAppModel
    @StateObject private var status = SandboxStatusModel()
    @State private var showLogs = false
    @State private var section = AppSection.status

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if model.embeddedDemoReady {
                StandaloneDemoView()
            } else if let url = model.webURL {
                if section == .status {
                    NativeDashboardView(status: status) {
                        Task { await status.refresh(baseURL: url) }
                    }
                } else {
                    SandboxWebView(
                        url: url,
                        onImportYOLO: { mapping in
                            model.importYOLODataset(canonicalToSourceID: mapping)
                        },
                        onInferImage: { experiment, comparison in
                            model.inferImage(
                                experimentID: experiment,
                                comparisonExperimentID: comparison
                            )
                        }
                    )
                }
            } else {
                setup
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .sheet(isPresented: $showLogs) { logSheet }
        .onChange(of: model.webURL) { url in
            if let url {
                section = .status
                status.start(baseURL: url)
            } else {
                status.stop()
            }
        }
        .onChange(of: model.profile) { _ in model.profileChanged() }
        .task { model.profileChanged() }
    }

    private var header: some View {
        HStack(spacing: 14) {
            Image(nsImage: NSApplication.shared.applicationIconImage)
                .resizable()
                .scaledToFit()
                .frame(width: 44, height: 44)
            VStack(alignment: .leading, spacing: 2) {
                Text("UAV Research Sandbox")
                    .font(.headline)
                Text(model.statusDetail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer()
            if model.webURL != nil && !model.embeddedDemoReady {
                Picker("Section", selection: $section) {
                    Label("Status", systemImage: "gauge.with.dots.needle.67percent")
                        .tag(AppSection.status)
                    Label("Workbench", systemImage: "rectangle.3.group")
                        .tag(AppSection.workbench)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 230)
            }
            Button {
                showLogs = true
            } label: {
                Label("Service Log", systemImage: "text.alignleft")
            }
            statusBadge
            if model.canStop {
                Button("Stop", role: .destructive) { model.stop() }
            }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
    }

    private var statusBadge: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
            Text(model.state.label)
                .font(.caption.weight(.semibold))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.quaternary, in: Capsule())
    }

    private var statusColor: Color {
        switch model.state {
        case .online, .connected: return .green
        case .failed: return .red
        case .preparing, .starting, .stopping: return .orange
        case .idle: return .secondary
        }
    }

    private var setup: some View {
        VStack(spacing: 28) {
            Spacer()
            Image(systemName: "shippingbox.and.arrow.backward")
                .font(.system(size: 54, weight: .light))
                .foregroundStyle(Color.accentColor)
            VStack(spacing: 8) {
                Text("Start the local Sandbox")
                    .font(.system(size: 28, weight: .semibold))
                Text(model.profile == .demo
                    ? "Run a deterministic ML example without installing Python, PX4, Gazebo, or the project repository."
                    : "The App manages the existing loopback service. Simulator and ML workflows remain in the Python project.")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 570)
            }
            VStack(alignment: .leading, spacing: 16) {
                HStack(spacing: 10) {
                    firstRunStep(1, "Profile", ready: true)
                    firstRunStep(
                        2, "Project",
                        ready: !model.profile.requiresProject || !model.projectRoot.isEmpty
                    )
                    firstRunStep(
                        3, "Dependencies",
                        ready: !model.profile.requiresProject || model.runtimeAssessment?.ready == true
                    )
                    firstRunStep(4, "Start", ready: model.canStart)
                }
                .accessibilityElement(children: .contain)
                Divider()
                if model.profile.requiresProject {
                    LabeledContent("Project") {
                        HStack {
                            Text(model.projectRoot.isEmpty ? "Not selected" : model.projectRoot)
                                .foregroundStyle(model.projectRoot.isEmpty ? .secondary : .primary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Button("Choose…") { model.chooseProject() }
                        }
                    }
                } else {
                    LabeledContent("Runtime") {
                        Label("Included in the App", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    }
                }
                LabeledContent("Profile") {
                    Picker("Profile", selection: $model.profile) {
                        ForEach(SandboxProfile.allCases) { profile in
                            Text(profile.displayName).tag(profile)
                        }
                    }
                    .labelsHidden()
                    .frame(width: 170)
                }
                if model.profile.requiresProject {
                    RuntimeCompatibilityView(model: model)
                }
                Divider()
                HStack {
                    Label("Loopback only · one managed job · bounded outputs", systemImage: "lock.shield")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Start Sandbox") { model.start() }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)
                        .disabled(!model.canStart)
                }
            }
            .padding(22)
            .frame(width: 680)
            .background(.background, in: RoundedRectangle(cornerRadius: 16))
            .overlay {
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
            }
            Spacer()
        }
        .padding(30)
    }

    private func firstRunStep(_ number: Int, _ title: String, ready: Bool) -> some View {
        HStack(spacing: 7) {
            Image(systemName: ready ? "checkmark.circle.fill" : "\(number).circle")
                .foregroundStyle(ready ? .green : .secondary)
            Text(title).font(.caption.weight(.medium))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var logSheet: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Sandbox service log").font(.title2.weight(.semibold))
                Spacer()
                Button("Done") { showLogs = false }
                    .keyboardShortcut(.cancelAction)
            }
            ScrollView {
                Text(model.logLines.isEmpty ? "No service output yet." : model.logLines.joined(separator: "\n"))
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
            }
            .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
        }
        .padding(20)
        .frame(minWidth: 720, minHeight: 460)
    }
}

private enum AppSection: Hashable {
    case status
    case workbench
}
