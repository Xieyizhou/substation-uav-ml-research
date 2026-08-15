// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "UAVSandboxApp",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "UAVSandboxApp", targets: ["UAVSandboxApp"]),
    ],
    targets: [
        .target(name: "SandboxAppCore"),
        .executableTarget(
            name: "UAVSandboxApp",
            dependencies: ["SandboxAppCore"]
        ),
        .testTarget(
            name: "SandboxAppCoreTests",
            dependencies: ["SandboxAppCore"]
        ),
    ],
    swiftLanguageModes: [.v5]
)
