// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "audiocapture",
    platforms: [
        .macOS(.v14)
    ],
    targets: [
        .executableTarget(
            name: "audiocapture",
            path: "Sources/audiocapture"
        )
    ]
)
