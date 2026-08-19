import Foundation

public enum SandboxProfile: String, CaseIterable, Identifiable, Sendable {
    case demo
    case development
    case formal

    public var id: String { rawValue }

    public var displayName: String {
        rawValue.capitalized
    }

    public var requiresProject: Bool {
        self != .demo
    }
}

public struct SandboxProject: Equatable, Sendable {
    public let root: URL
    public let runtime: VerifiedRuntimeProfile

    public init(root: URL, runtime: VerifiedRuntimeProfile) {
        self.root = root
        self.runtime = runtime
    }

    public var python: URL {
        URL(fileURLWithPath: runtime.pythonExecutable)
    }

    public var mainScript: URL {
        root.appendingPathComponent("main.py")
    }

    public func runtimeEnvironment(
        base: [String: String] = ProcessInfo.processInfo.environment
    ) -> [String: String] {
        var environment = base
        let inherited = (base["PATH"] ?? "/usr/bin:/bin")
            .split(separator: ":")
            .map(String.init)
        let verified = [
            python.deletingLastPathComponent().path,
            URL(fileURLWithPath: runtime.gazeboExecutable).deletingLastPathComponent().path,
            URL(fileURLWithPath: runtime.qtPrefix).appendingPathComponent("bin").path,
            URL(fileURLWithPath: runtime.openCVPrefix).appendingPathComponent("bin").path,
        ]
        let fallback = [
            "/opt/homebrew/bin", "/opt/homebrew/sbin",
            "/usr/local/bin", "/usr/local/sbin",
        ]
        var seen = Set<String>()
        let folders = (verified + fallback + inherited).filter {
            !$0.isEmpty && seen.insert($0).inserted
        }
        environment["PATH"] = folders.joined(separator: ":")
        environment["UAV_SANDBOX_PROJECT_ROOT"] = root.path
        environment["UAV_SANDBOX_PYTHON"] = runtime.pythonExecutable
        environment["UAV_SANDBOX_GZ_EXECUTABLE"] = runtime.gazeboExecutable
        environment["PX4_ROOT"] = runtime.px4Root
        environment["OpenCV_DIR"] = URL(fileURLWithPath: runtime.openCVPrefix)
            .appendingPathComponent("lib/cmake/opencv4").path
        environment["Qt5_DIR"] = URL(fileURLWithPath: runtime.qtPrefix)
            .appendingPathComponent("lib/cmake/Qt5").path
        environment["UAV_SANDBOX_OPENCV_PREFIX"] = runtime.openCVPrefix
        environment["UAV_SANDBOX_QT_PREFIX"] = runtime.qtPrefix
        environment["UAV_SANDBOX_RUNTIME_PROFILE"] = RuntimeProfileStore().fileURL.path
        environment["MPLCONFIGDIR"] = root
            .appendingPathComponent("outputs/sandbox/cache/matplotlib").path
        return environment
    }

    public func arguments(
        profile: SandboxProfile,
        port: UInt16,
        command: String
    ) -> [String] {
        var values = [
            mainScript.path,
            "sandbox",
            "--project-root", root.path,
            "--profile", profile.rawValue,
            command,
        ]
        if command == "serve" {
            values.append(contentsOf: [
                "--host", "127.0.0.1",
                "--port", String(port),
            ])
        }
        return values
    }
}

public enum ProjectValidationError: LocalizedError, Equatable {
    case missingMainScript
    case missingSandboxPackage

    public var errorDescription: String? {
        switch self {
        case .missingMainScript:
            return "The selected folder does not contain main.py."
        case .missingSandboxPackage:
            return "The selected folder does not contain src/sandbox."
        }
    }
}

public enum ProjectLocator {
    public static func locate(
        root: URL,
        runtime: VerifiedRuntimeProfile,
        fileManager: FileManager = .default
    ) throws -> SandboxProject {
        let normalized = try validate(root: root, fileManager: fileManager)
        guard URL(fileURLWithPath: runtime.projectRoot).standardizedFileURL == normalized
        else { throw ProjectValidationError.missingMainScript }
        return SandboxProject(root: normalized, runtime: runtime)
    }

    public static func validate(
        root: URL,
        fileManager: FileManager = .default
    ) throws -> URL {
        let normalized = root.standardizedFileURL
        guard fileManager.fileExists(
            atPath: normalized.appendingPathComponent("main.py").path
        ) else {
            throw ProjectValidationError.missingMainScript
        }
        guard fileManager.fileExists(
            atPath: normalized.appendingPathComponent("src/sandbox").path
        ) else {
            throw ProjectValidationError.missingSandboxPackage
        }
        return normalized
    }

    public static func suggestedRoot(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        currentDirectory: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    ) -> URL? {
        if let configured = environment["UAV_SANDBOX_PROJECT_ROOT"], !configured.isEmpty {
            return URL(fileURLWithPath: configured).standardizedFileURL
        }
        var candidate = currentDirectory.standardizedFileURL
        for _ in 0..<6 {
            if FileManager.default.fileExists(
                atPath: candidate.appendingPathComponent("main.py").path
            ) {
                return candidate
            }
            candidate.deleteLastPathComponent()
        }
        return nil
    }

}
