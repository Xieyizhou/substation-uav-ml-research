import Foundation

public enum SandboxProfile: String, CaseIterable, Identifiable, Sendable {
    case demo
    case development
    case formal

    public var id: String { rawValue }

    public var displayName: String {
        rawValue.capitalized
    }
}

public struct SandboxProject: Equatable, Sendable {
    public let root: URL
    public let python: URL

    public init(root: URL, python: URL) {
        self.root = root
        self.python = python
    }

    public var mainScript: URL {
        root.appendingPathComponent("main.py")
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
    case missingPython

    public var errorDescription: String? {
        switch self {
        case .missingMainScript:
            return "The selected folder does not contain main.py."
        case .missingSandboxPackage:
            return "The selected folder does not contain src/sandbox."
        case .missingPython:
            return "No usable Python 3 executable was found."
        }
    }
}

public enum ProjectLocator {
    public static func locate(
        root: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) throws -> SandboxProject {
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
        guard let python = pythonURL(
            root: normalized,
            environment: environment,
            fileManager: fileManager
        ) else {
            throw ProjectValidationError.missingPython
        }
        return SandboxProject(root: normalized, python: python)
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

    private static func pythonURL(
        root: URL,
        environment: [String: String],
        fileManager: FileManager
    ) -> URL? {
        let virtualEnvironment = root.appendingPathComponent(".venv/bin/python")
        if fileManager.isExecutableFile(atPath: virtualEnvironment.path) {
            return virtualEnvironment
        }
        let searchPath = environment["PATH"] ?? "/usr/local/bin:/usr/bin:/bin"
        for name in ["python3", "python"] {
            for folder in searchPath.split(separator: ":") {
                let candidate = URL(fileURLWithPath: String(folder))
                    .appendingPathComponent(name)
                if fileManager.isExecutableFile(atPath: candidate.path) {
                    return candidate
                }
            }
        }
        return nil
    }
}
