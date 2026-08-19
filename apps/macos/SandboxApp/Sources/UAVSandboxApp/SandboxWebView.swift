import SandboxAppCore
import SwiftUI
import WebKit

struct SandboxWebView: NSViewRepresentable {
    let url: URL
    let onImportYOLO: ([String: Int]) -> Void
    let onInferImage: (String, String?) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(onImportYOLO: onImportYOLO, onInferImage: onInferImage)
    }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        let view = WKWebView(frame: .zero, configuration: configuration)
        view.navigationDelegate = context.coordinator
        view.allowsMagnification = true
        view.load(URLRequest(url: url))
        return view
    }

    func updateNSView(_ view: WKWebView, context: Context) {
        guard !LocalWebDocument.isSame(current: view.url, target: url) else { return }
        view.load(URLRequest(url: url))
    }

    final class Coordinator: NSObject, WKNavigationDelegate {
        let onImportYOLO: ([String: Int]) -> Void
        let onInferImage: (String, String?) -> Void

        init(
            onImportYOLO: @escaping ([String: Int]) -> Void,
            onInferImage: @escaping (String, String?) -> Void
        ) {
            self.onImportYOLO = onImportYOLO
            self.onInferImage = onInferImage
        }

        func webView(
            _ webView: WKWebView,
            decidePolicyFor navigationAction: WKNavigationAction,
            decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
        ) {
            if navigationAction.request.url?.scheme == "uav-sandbox",
               navigationAction.request.url?.host == "import-yolo" {
                let components = URLComponents(
                    url: navigationAction.request.url!, resolvingAgainstBaseURL: false
                )
                let allowed = Set(["transformer", "switchgear", "capacitor_bank", "reactor"])
                let mapping = Dictionary(uniqueKeysWithValues: (components?.queryItems ?? [])
                    .compactMap { item -> (String, Int)? in
                        guard allowed.contains(item.name), let value = item.value,
                              let classID = Int(value), classID >= 0 else { return nil }
                        return (item.name, classID)
                    })
                onImportYOLO(mapping)
                decisionHandler(.cancel)
                return
            }
            if navigationAction.request.url?.scheme == "uav-sandbox",
               navigationAction.request.url?.host == "infer-image" {
                let components = URLComponents(
                    url: navigationAction.request.url!, resolvingAgainstBaseURL: false
                )
                let values = Dictionary(uniqueKeysWithValues: (components?.queryItems ?? [])
                    .compactMap { item in item.value.map { (item.name, $0) } })
                if let experiment = values["experiment_id"], !experiment.isEmpty {
                    onInferImage(experiment, values["comparison_experiment_id"])
                }
                decisionHandler(.cancel)
                return
            }
            guard let host = navigationAction.request.url?.host,
                  ["127.0.0.1", "localhost", "::1"].contains(host) else {
                decisionHandler(.cancel)
                return
            }
            decisionHandler(.allow)
        }
    }
}
