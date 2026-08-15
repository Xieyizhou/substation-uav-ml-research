import SandboxAppCore
import SwiftUI

struct RuntimeCandidateView: View {
    @ObservedObject var model: SandboxAppModel
    @Binding var isPresented: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Runtime candidates").font(.title2.weight(.semibold))
                    Text("Select a compatible installation, then revalidate before launch.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if model.runtimeRefreshing { ProgressView().controlSize(.small) }
                Button("Done") { isPresented = false }
                    .keyboardShortcut(.cancelAction)
            }
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(model.runtimeCandidates) { group in
                        candidateGroup(group)
                    }
                }
                .padding(.vertical, 2)
            }
            Divider()
            HStack(spacing: 8) {
                Menu("Choose manually") {
                    Button("Python executable…") { model.choosePython() }
                    Button("PX4 checkout…") { model.choosePX4() }
                    Button("Gazebo executable…") { model.chooseGazebo() }
                    Button("OpenCV 4 prefix…") { model.chooseOpenCV() }
                    Button("Qt 5 prefix…") { model.chooseQt() }
                }
                Button("Use automatic priority") { model.resetRuntimeSelection() }
                Spacer()
                Button("Refresh") { model.validateRuntime() }
                    .disabled(model.runtimeRefreshing)
            }
            .controlSize(.small)
        }
        .padding(20)
        .frame(minWidth: 760, minHeight: 540)
    }

    private func candidateGroup(_ group: RuntimeCandidateGroup) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(group.title).font(.headline)
            if group.candidates.isEmpty {
                Text("No candidates discovered.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(group.candidates) { candidate in
                    candidateRow(candidate)
                }
            }
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor),
                    in: RoundedRectangle(cornerRadius: 10))
    }

    private func candidateRow(_ candidate: RuntimeCandidate) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon(candidate.status))
                .foregroundStyle(color(candidate.status))
                .frame(width: 16)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(candidate.version ?? "Unknown version")
                        .font(.caption.weight(.semibold))
                    Text(candidate.origin.rawValue)
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.quaternary, in: Capsule())
                }
                Text(candidate.path)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .lineLimit(2)
                    .truncationMode(.middle)
                Text(candidate.detail)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            Spacer()
            if candidate.selected {
                Label("Selected", systemImage: "checkmark")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.green)
            } else {
                Button("Use") { model.selectRuntimeCandidate(candidate) }
                    .controlSize(.small)
                    .disabled(!selectable(candidate.status) || model.runtimeRefreshing)
            }
        }
    }

    private func selectable(_ status: RuntimeCompatibilityStatus) -> Bool {
        status == .compatible || status == .compatibleWithWarning || status == .untested
    }

    private func color(_ status: RuntimeCompatibilityStatus) -> Color {
        switch status {
        case .compatible: return .green
        case .compatibleWithWarning, .untested: return .orange
        case .missing, .unsupported, .changedSinceValidation: return .red
        }
    }

    private func icon(_ status: RuntimeCompatibilityStatus) -> String {
        switch status {
        case .compatible: return "checkmark.circle.fill"
        case .compatibleWithWarning, .untested: return "exclamationmark.triangle.fill"
        case .missing, .unsupported, .changedSinceValidation: return "xmark.circle.fill"
        }
    }
}
