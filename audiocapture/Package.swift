// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "audiocapture",
    platforms: [
        .macOS(.v13)  // ScreenCaptureKit requires macOS 13+
    ],
    targets: [
        .executableTarget(
            name: "audiocapture",
            path: "Sources/audiocapture"
        )
    ]
)
