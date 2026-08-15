import SandboxAppCore
import SwiftUI

struct StandaloneDemoView: View {
    @State private var result: StandaloneDemoResult?
    @State private var failure: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                heading
                if let result {
                    metrics(result)
                    predictions(result)
                    artifact(result)
                } else if let failure {
                    failureCard(failure)
                } else {
                    ProgressView("Running the built-in classifier…")
                        .frame(maxWidth: .infinity, minHeight: 240)
                }
            }
            .padding(30)
            .frame(maxWidth: 1120, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .task { if result == nil { run() } }
    }

    private var heading: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 7) {
                Text("Built-in ML Demo")
                    .font(.system(size: 30, weight: .bold, design: .rounded))
                Text("A deterministic nearest-centroid safety classifier running entirely inside the App.")
                    .foregroundStyle(.secondary)
                Label(
                    "Synthetic demonstration only · no repository, Python, PX4, or Gazebo required",
                    systemImage: "shippingbox.fill"
                )
                .font(.caption.weight(.semibold))
                .foregroundStyle(.teal)
            }
            Spacer()
            Button("Run again", action: run)
                .buttonStyle(.borderedProminent)
        }
    }

    private func metrics(_ value: StandaloneDemoResult) -> some View {
        HStack(spacing: 14) {
            demoMetric("Accuracy", value: value.accuracy.formatted(.percent.precision(.fractionLength(1))))
            demoMetric("Macro F1", value: value.macroF1.formatted(.number.precision(.fractionLength(3))))
            demoMetric("Evaluation", value: "\(value.evaluationSampleCount) samples")
            demoMetric("Gate", value: value.passed ? "Passed" : "Failed", tint: value.passed ? .green : .red)
        }
    }

    private func demoMetric(
        _ title: String,
        value: String,
        tint: Color = .teal
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.caption2.weight(.bold))
                .foregroundStyle(.secondary)
            Text(value).font(.title2.weight(.bold)).lineLimit(1)
        }
        .padding(17)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(tint.opacity(0.09), in: RoundedRectangle(cornerRadius: 14))
        .overlay { RoundedRectangle(cornerRadius: 14).stroke(tint.opacity(0.25)) }
    }

    private func predictions(_ value: StandaloneDemoResult) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Evaluation predictions").font(.title3.weight(.semibold))
            Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 9) {
                GridRow {
                    tableHeader("Sample")
                    tableHeader("Distance")
                    tableHeader("Density")
                    tableHeader("Expected")
                    tableHeader("Predicted")
                    tableHeader("Result")
                }
                Divider().gridCellColumns(6)
                ForEach(value.predictions) { item in
                    GridRow {
                        Text(item.sampleID).font(.caption.monospaced())
                        Text("\(item.minimumDistanceM, specifier: "%.1f") m")
                        Text(item.obstacleDensity, format: .number.precision(.fractionLength(2)))
                        Text(item.expected.capitalized)
                        Text(item.predicted.capitalized)
                        Image(systemName: item.expected == item.predicted ? "checkmark.circle.fill" : "xmark.circle.fill")
                            .foregroundStyle(item.expected == item.predicted ? .green : .red)
                    }
                }
            }
            .font(.subheadline)
        }
        .padding(20)
        .background(.background, in: RoundedRectangle(cornerRadius: 14))
        .overlay { RoundedRectangle(cornerRadius: 14).stroke(.separator) }
    }

    private func artifact(_ value: StandaloneDemoResult) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Label("Reproducible artifact", systemImage: "checkmark.seal")
                .font(.headline)
            Text(value.scopeNote).foregroundStyle(.secondary)
            Text(value.artifactIdentitySHA256)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background, in: RoundedRectangle(cornerRadius: 14))
        .overlay { RoundedRectangle(cornerRadius: 14).stroke(.separator) }
    }

    private func tableHeader(_ value: String) -> some View {
        Text(value).font(.caption.weight(.bold)).foregroundStyle(.secondary)
    }

    private func failureCard(_ message: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 38, weight: .light))
                .foregroundStyle(.orange)
            Text("Demo failed").font(.title2.weight(.semibold))
            Text(message).foregroundStyle(.secondary)
            Button("Try again", action: run)
        }
        .frame(maxWidth: .infinity, minHeight: 260)
    }

    private func run() {
        do {
            failure = nil
            result = try StandaloneDemo.run(outputDirectory: outputDirectory)
        } catch {
            result = nil
            failure = error.localizedDescription
        }
    }

    private var outputDirectory: URL {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first!
        return base
            .appendingPathComponent("UAV Research Sandbox", isDirectory: true)
            .appendingPathComponent("Demo", isDirectory: true)
    }
}
