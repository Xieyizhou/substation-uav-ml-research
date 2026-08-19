import Foundation

public enum LocalWebDocument {
    public static func isSame(current: URL?, target: URL) -> Bool {
        guard let current,
              var currentParts = URLComponents(url: current, resolvingAgainstBaseURL: false),
              var targetParts = URLComponents(url: target, resolvingAgainstBaseURL: false)
        else { return false }

        normalize(&currentParts)
        normalize(&targetParts)
        return currentParts == targetParts
    }

    private static func normalize(_ parts: inout URLComponents) {
        parts.scheme = parts.scheme?.lowercased()
        parts.host = parts.host?.lowercased()
        if parts.path.isEmpty { parts.path = "/" }
        parts.fragment = nil
    }
}
