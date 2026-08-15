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
            if let url = model.webURL {
                if section == .status {
                    NativeDashboardView(status: status) {
                        Task { await status.refresh(baseURL: url) }
                    }
                } else {
                    SandboxWebView(url: url)
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
        .onReceive(NotificationCenter.default.publisher(
            for: NSApplication.willTerminateNotification
        )) { _ in
            model.shutdownBeforeApplicationExit()
        }
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
            if model.webURL != nil {
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
                Text("The App manages the existing loopback service. Simulator and ML workflows remain in the Python project.")
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 570)
            }
            VStack(alignment: .leading, spacing: 16) {
                LabeledContent("Project") {
                    HStack {
                        Text(model.projectRoot.isEmpty ? "Not selected" : model.projectRoot)
                            .foregroundStyle(model.projectRoot.isEmpty ? .secondary : .primary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Button("Choose…") { model.chooseProject() }
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
