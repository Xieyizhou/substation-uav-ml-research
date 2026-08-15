import Foundation

public struct RuntimeConflict: Equatable, Sendable {
    public let process: String
    public let pid: Int
    public let detail: String
}

public struct RuntimeConflictChecker: Sendable {
    private let runner: any RuntimeCommandRunning

    public init(runner: any RuntimeCommandRunning = LocalRuntimeCommandRunner()) {
        self.runner = runner
    }

    public func conflicts() -> [RuntimeConflict] {
        let ps = URL(fileURLWithPath: "/bin/ps")
        let root = URL(fileURLWithPath: "/")
        guard let result = try? runner.run(ps, ["-axo", "pid=,comm=,args="], at: root),
              result.exitCode == 0 else { return [] }
        return result.output.split(whereSeparator: \.isNewline).compactMap(parse)
    }

    private func parse(_ line: Substring) -> RuntimeConflict? {
        let fields = line.trimmingCharacters(in: .whitespaces).split(
            maxSplits: 2, whereSeparator: \.isWhitespace
        )
        guard fields.count == 3, let pid = Int(fields[0]) else { return nil }
        let executable = URL(fileURLWithPath: String(fields[1])).lastPathComponent.lowercased()
        let command = String(fields[2])
        if executable == "px4" {
            return RuntimeConflict(process: "PX4", pid: pid, detail: command)
        }
        if executable == "gz" && command.split(separator: " ").contains("sim") {
            return RuntimeConflict(process: "Gazebo", pid: pid, detail: command)
        }
        if ["gazebo", "gzserver"].contains(executable) {
            return RuntimeConflict(process: "Gazebo", pid: pid, detail: command)
        }
        return nil
    }
}
