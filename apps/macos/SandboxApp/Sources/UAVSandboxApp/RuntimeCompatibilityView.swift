import SandboxAppCore
import SwiftUI

struct RuntimeCompatibilityView: View {
    @ObservedObject var model: SandboxAppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Runtime compatibility", systemImage: "checkmark.shield")
                    .font(.headline)
                Spacer()
                Text(model.runtimeSummary)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(summaryColor)
            }
            if let assessment = model.runtimeAssessment {
                ForEach(assessment.components) { item in
                    component(item)
                }
                if !assessment.blockers.isEmpty {
                    Text(assessment.blockers.joined(separator: "\n"))
                        .font(.caption)
                        .foregroundStyle(.red)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } else {
                Text(model.projectRoot.isEmpty
                     ? "Choose the project before validating external runtimes."
                     : "Select Check runtime to discover compatible installations.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 8) {
                Button("Python…") { model.choosePython() }
                Button("PX4…") { model.choosePX4() }
                Button("Gazebo…") { model.chooseGazebo() }
                Spacer()
                if model.runtimeRefreshing { ProgressView().controlSize(.small) }
                Button("Check runtime") { model.validateRuntime() }
                    .disabled(model.projectRoot.isEmpty || model.runtimeRefreshing)
            }
            .controlSize(.small)
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }

    private func component(_ item: RuntimeComponent) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: icon(item.status))
                .foregroundStyle(color(item.status))
                .frame(width: 14)
            VStack(alignment: .leading, spacing: 1) {
                Text(item.title).font(.caption.weight(.semibold))
                Text(item.detail)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .truncationMode(.middle)
            }
            Spacer()
            Text(item.status.rawValue.replacingOccurrences(of: "_", with: " "))
                .font(.caption2.weight(.medium))
                .foregroundStyle(color(item.status))
        }
    }

    private var summaryColor: Color {
        guard let value = model.runtimeAssessment?.overall else { return .secondary }
        return color(value)
    }

    private func color(_ status: RuntimeCompatibilityStatus) -> Color {
        switch status {
        case .compatible: return .green
        case .compatibleWithWarning, .untested: return .orange
        case .unsupported, .missing, .changedSinceValidation: return .red
        }
    }

    private func icon(_ status: RuntimeCompatibilityStatus) -> String {
        switch status {
        case .compatible: return "checkmark.circle.fill"
        case .compatibleWithWarning, .untested: return "exclamationmark.triangle.fill"
        case .unsupported, .missing, .changedSinceValidation: return "xmark.circle.fill"
        }
    }
}
