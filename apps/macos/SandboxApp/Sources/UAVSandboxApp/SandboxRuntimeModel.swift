import AppKit
import Foundation
import SandboxAppCore

extension SandboxAppModel {
    var runtimeSummary: String {
        guard profile != .demo else { return "Included in the App" }
        guard let runtimeAssessment else { return "Not validated" }
        switch runtimeAssessment.overall {
        case .compatible: return "Ready"
        case .compatibleWithWarning, .untested: return "Ready with warnings"
        default: return "Blocked"
        }
    }

    func profileChanged() {
        if profile == .demo {
            runtimeAssessment = nil
            return
        }
        Task { await refreshRuntime() }
    }

    func refreshRuntime(persist: Bool = false) async {
        guard profile != .demo, !projectRoot.isEmpty else {
            runtimeAssessment = nil
            return
        }
        runtimeRefreshing = true
        defer { runtimeRefreshing = false }
        let root = URL(fileURLWithPath: projectRoot)
        let selectedProfile = profile
        let selection = runtimeSelection
        let manager = runtimeManager
        do {
            let assessment = try await Task.detached {
                if persist {
                    return try manager.validateAndPersist(
                        projectRoot: root, profile: selectedProfile,
                        selection: selection
                    )
                }
                return try manager.assess(
                    projectRoot: root, profile: selectedProfile,
                    selection: selection
                )
            }.value
            runtimeAssessment = assessment
            if persist, assessment.ready {
                append("Saved verified runtime profile at \(manager.store.fileURL.path).")
            }
        } catch {
            runtimeAssessment = RuntimeAssessment(
                components: [], overall: .missing,
                blockers: [error.localizedDescription], selected: nil
            )
        }
    }

    func validateRuntime() {
        Task { await refreshRuntime(persist: true) }
    }

    func choosePython() {
        chooseFile(title: "Choose a Python executable") { url in
            self.runtimeSelection.pythonExecutable = url
        }
    }

    func choosePX4() {
        chooseDirectory(title: "Choose the PX4-Autopilot checkout") { url in
            self.runtimeSelection.px4Root = url
        }
    }

    func chooseGazebo() {
        chooseFile(title: "Choose the gz executable") { url in
            self.runtimeSelection.gazeboExecutable = url
        }
    }

    private func chooseFile(title: String, update: (URL) -> Void) {
        let panel = NSOpenPanel()
        panel.title = title
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            update(url.standardizedFileURL)
            validateRuntime()
        }
    }

    private func chooseDirectory(title: String, update: (URL) -> Void) {
        let panel = NSOpenPanel()
        panel.title = title
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            update(url.standardizedFileURL)
            validateRuntime()
        }
    }

    func runtimeForLaunch(profile: SandboxProfile) async throws -> VerifiedRuntimeProfile {
        let root = URL(fileURLWithPath: projectRoot)
        let manager = runtimeManager
        let selection = runtimeSelection
        let assessment = try await Task.detached {
            try manager.assess(
                projectRoot: root, profile: profile,
                selection: selection
            )
        }.value
        runtimeAssessment = assessment
        guard assessment.ready, let selected = assessment.selected else {
            throw AppFailure.runtimeCompatibility(assessment.blockers)
        }
        let storedRoot = manager.store.load().map {
            URL(fileURLWithPath: $0.projectRoot).standardizedFileURL
        }
        if storedRoot != root.standardizedFileURL {
            let saved = try await Task.detached {
                try manager.validateAndPersist(
                    projectRoot: root, profile: profile,
                    selection: selection
                )
            }.value
            guard let selected = saved.selected else {
                throw AppFailure.runtimeCompatibility(saved.blockers)
            }
            runtimeAssessment = saved
            return selected
        }
        return selected
    }
}
