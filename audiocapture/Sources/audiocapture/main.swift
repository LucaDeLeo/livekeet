@preconcurrency import AVFoundation
import Accelerate
import CoreMedia
import Darwin
import Foundation
import ScreenCaptureKit

// MARK: - Configuration

let targetSampleRate: Double = 16000
let outputFrameSize = 1024  // ~64ms at 16kHz - larger frames = more timing tolerance
let ringBufferCapacity = 32768  // Power-of-2 for bitmask wrapping (~2s at 16kHz)
let minBufferSamples = Int(targetSampleRate * 0.2)  // 200ms pre-buffer before starting output
let outputChannels = 2  // Stereo: left=mic, right=system (for speaker detection)

// MARK: - CMSampleBuffer Extension

extension CMSampleBuffer {
    var asPCMBuffer: AVAudioPCMBuffer? {
        try? withAudioBufferList { audioBufferList, _ in
            guard let desc = formatDescription?.audioStreamBasicDescription,
                  let format = AVAudioFormat(standardFormatWithSampleRate: desc.mSampleRate, channels: desc.mChannelsPerFrame) else {
                return nil
            }
            return AVAudioPCMBuffer(pcmFormat: format, bufferListNoCopy: audioBufferList.unsafePointer)
        }
    }
}

// MARK: - Lock-Free Ring Buffer (OBS-style, no runtime allocations)

final class RingBuffer: @unchecked Sendable {
    private let buffer: UnsafeMutablePointer<Float>
    private let capacity: Int
    private let mask: Int
    private var writeIndex: Int = 0
    private var readIndex: Int = 0
    private var availableCount: Int = 0
    private var lock = os_unfair_lock()
    private var underrunCount: Int = 0

    init(capacity: Int) {
        precondition(capacity & (capacity - 1) == 0, "Ring buffer capacity must be a power of 2")
        self.capacity = capacity
        self.mask = capacity - 1
        self.buffer = .allocate(capacity: capacity)
        self.buffer.initialize(repeating: 0, count: capacity)
    }

    deinit {
        buffer.deallocate()
    }

    /// Write samples to the ring buffer. No allocations during write.
    func write(_ samples: UnsafePointer<Float>, count: Int) {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        // Don't overflow - drop oldest data if necessary
        let toWrite = min(count, capacity)
        let overflow = (availableCount + toWrite) - capacity
        if overflow > 0 {
            // Advance read pointer to make room
            readIndex = (readIndex + overflow) & mask
            availableCount -= overflow
        }

        // Write samples with memcpy (two-part for wraparound)
        let firstPart = min(toWrite, capacity - writeIndex)
        buffer.advanced(by: writeIndex).update(from: samples, count: firstPart)
        if firstPart < toWrite {
            buffer.update(from: samples.advanced(by: firstPart), count: toWrite - firstPart)
        }
        writeIndex = (writeIndex + toWrite) & mask
        availableCount += toWrite
    }

    /// Write samples from an array
    func write(_ samples: [Float]) {
        samples.withUnsafeBufferPointer { ptr in
            if let base = ptr.baseAddress {
                write(base, count: samples.count)
            }
        }
    }

    /// Read samples into an Int16 output buffer. Fills with silence on underrun.
    /// Returns the number of actual samples read (rest is silence).
    func read(into output: UnsafeMutablePointer<Int16>, count: Int) -> Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        let toRead = min(count, availableCount)

        // Convert float samples to Int16 with clipping
        for i in 0..<toRead {
            let sample = buffer[(readIndex + i) & mask]
            let clamped = max(-1.0, min(1.0, sample))
            output[i] = Int16(clamped * 32767)
        }

        // Fill remainder with silence if underrun
        if toRead < count {
            for i in toRead..<count {
                output[i] = 0
            }
            if toRead > 0 {
                underrunCount += 1
            }
        }

        readIndex = (readIndex + toRead) & mask
        availableCount -= toRead
        return toRead
    }

    /// Read samples into a Float output buffer. Fills with silence on underrun.
    func readFloat(into output: UnsafeMutablePointer<Float>, count: Int) -> Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        let toRead = min(count, availableCount)

        // Read with memcpy (two-part for wraparound)
        let firstPart = min(toRead, capacity - readIndex)
        output.update(from: buffer.advanced(by: readIndex), count: firstPart)
        if firstPart < toRead {
            output.advanced(by: firstPart).update(from: buffer, count: toRead - firstPart)
        }

        // Fill remainder with silence if underrun
        if toRead < count {
            for i in toRead..<count {
                output[i] = 0
            }
            if toRead > 0 {
                underrunCount += 1
            }
        }

        readIndex = (readIndex + toRead) & mask
        availableCount -= toRead
        return toRead
    }

    var available: Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return availableCount
    }

    var underruns: Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return underrunCount
    }
}

// MARK: - Audio Capture Manager

final class AudioCaptureManager: @unchecked Sendable {
    private var stream: SCStream?
    private var micEngine: AVAudioEngine?
    private var outputHandler: StreamOutputHandler?
    private var isRunning = false
    private var outputTimer: DispatchSourceTimer?

    // Separate buffers for proper mixing
    let systemBuffer = RingBuffer(capacity: ringBufferCapacity)
    let micBuffer = RingBuffer(capacity: ringBufferCapacity)
    private let outputQueue = DispatchQueue(label: "audiocapture.output", qos: .userInteractive)
    private let audioSampleQueue = DispatchQueue(label: "audiocapture.sck-audio", qos: .userInteractive)
    private let screenDropQueue = DispatchQueue(label: "audiocapture.screen-drop", qos: .background)

    private var systemAudioConverter: AVAudioConverter?
    private let outputFormat: AVAudioFormat

    var includeMic: Bool = true
    var micGain: Float = 1.0

    // Thread-safe converter access
    private var converterLock = os_unfair_lock()

    // System audio health tracking
    private var systemAudioFrameCount: UInt64 = 0

    // PTS gap detection (accessed only on serial audioSampleQueue)
    private var lastSystemPTS: CMTime = .invalid
    private var lastSystemDuration: CMTime = .invalid

    // Stream restart state (protected by restartState lock)
    private struct RestartState {
        var count = 0
        var isRestarting = false
        var lastTime: Date?
        var lastAttemptTime: Date?
    }
    private let maxRestarts = 20
    private let minRestartInterval: TimeInterval = 3
    private let restartState = OSAllocatedUnfairLock(initialState: RestartState())

    init() {
        outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: targetSampleRate, channels: 1, interleaved: false)!
    }

    // MARK: - List Devices

    func listDevices() {
        let deviceTypes: [AVCaptureDevice.DeviceType] = [.microphone, .external]
        let devices = AVCaptureDevice.DiscoverySession(deviceTypes: deviceTypes, mediaType: .audio, position: .unspecified).devices
        fputs("Available microphones:\n", stderr)
        for device in devices {
            let isDefault = device.uniqueID == AVCaptureDevice.default(for: .audio)?.uniqueID
            fputs("  \(device.localizedName)\(isDefault ? " (default)" : "")\n", stderr)
        }
    }

    // MARK: - Permissions

    func requestMicrophonePermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .audio)
        default: return false
        }
    }

    // MARK: - Stream Configuration

    private func makeStreamConfig() -> SCStreamConfiguration {
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 48000
        config.channelCount = 2
        config.queueDepth = 8
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        config.showsCursor = false
        return config
    }

    // MARK: - Start Capture

    func startCapture() async throws {
        if includeMic {
            let hasPermission = await requestMicrophonePermission()
            if !hasPermission {
                fputs("Warning: Microphone permission not granted\n", stderr)
            }
        }

        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else {
            throw CaptureError.noDisplay
        }

        let config = makeStreamConfig()
        let filter = SCContentFilter(display: display, excludingWindows: [])

        outputHandler = StreamOutputHandler(manager: self)
        stream = SCStream(filter: filter, configuration: config, delegate: outputHandler)
        try stream?.addStreamOutput(outputHandler!, type: .audio, sampleHandlerQueue: audioSampleQueue)
        try stream?.addStreamOutput(outputHandler!, type: .screen, sampleHandlerQueue: screenDropQueue)

        if includeMic { try startMicrophoneCapture() }

        try await stream?.startCapture()
        isRunning = true

        fputs("Capture started. Press Ctrl+C to stop.\n", stderr)
        startOutputLoop()
    }

    // MARK: - Audio Conversion Helper

    private func convertAndExtractSamples(
        from inputBuffer: AVAudioPCMBuffer,
        using converter: AVAudioConverter,
        gain: Float = 1.0
    ) -> [Float]? {
        let ratio = targetSampleRate / inputBuffer.format.sampleRate
        let outputFrameCount = AVAudioFrameCount(Double(inputBuffer.frameLength) * ratio)

        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputFrameCount) else {
            return nil
        }

        var error: NSError?
        nonisolated(unsafe) var consumed = false
        let status = converter.convert(to: outputBuffer, error: &error) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return inputBuffer
        }

        guard status != .error, error == nil,
              let floatData = outputBuffer.floatChannelData else {
            return nil
        }

        let count = Int(outputBuffer.frameLength)
        let ptr = floatData[0]

        if gain == 1.0 {
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        }

        // Use vDSP for SIMD-optimized gain multiplication
        var result = [Float](repeating: 0, count: count)
        var gainValue = gain
        vDSP_vsmul(ptr, 1, &gainValue, &result, 1, vDSP_Length(count))
        return result
    }

    // MARK: - Process System Audio

    func processSystemAudio(_ sampleBuffer: CMSampleBuffer) {
        guard let pcmBuffer = sampleBuffer.asPCMBuffer else { return }

        // Track PTS for gap detection (thread-safe: runs on serial audioSampleQueue)
        let currentPTS = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let currentDuration = CMSampleBufferGetDuration(sampleBuffer)
        if lastSystemPTS.isValid && currentPTS.isValid {
            let expectedPTS = CMTimeAdd(lastSystemPTS, lastSystemDuration)
            let gap = CMTimeSubtract(currentPTS, expectedPTS)
            let gapSeconds = CMTimeGetSeconds(gap)
            if gapSeconds > 1.0 {
                fputs("Warning: system audio gap (\(String(format: "%.0f", gapSeconds * 1000))ms), restarting stream\n", stderr)
                restartCapture()
            } else if gapSeconds > 0.2 {
                fputs("Warning: system audio gap (\(String(format: "%.0f", gapSeconds * 1000))ms)\n", stderr)
            }
        }
        lastSystemPTS = currentPTS
        lastSystemDuration = currentDuration

        os_unfair_lock_lock(&converterLock)

        // Detect format changes (e.g. audio routing switch) and recreate converter
        if let existing = systemAudioConverter, existing.inputFormat != pcmBuffer.format {
            fputs("System audio format changed, recreating converter\n", stderr)
            systemAudioConverter = nil
        }

        if systemAudioConverter == nil {
            guard let converter = AVAudioConverter(from: pcmBuffer.format, to: outputFormat) else {
                os_unfair_lock_unlock(&converterLock)
                fputs("Warning: Could not create system audio converter\n", stderr)
                return
            }
            systemAudioConverter = converter
            fputs("System audio: \(pcmBuffer.format.sampleRate)Hz, \(pcmBuffer.format.channelCount)ch\n", stderr)
        }

        let converter = systemAudioConverter!
        os_unfair_lock_unlock(&converterLock)

        guard let samples = convertAndExtractSamples(from: pcmBuffer, using: converter) else {
            return
        }
        systemBuffer.write(samples)
        systemAudioFrameCount += 1

        // Reset restart counter after 30s of stable audio
        restartState.withLock { state in
            if let lastRestart = state.lastTime,
               Date().timeIntervalSince(lastRestart) > 30 {
                state.count = 0
                state.lastTime = nil
            }
        }
    }

    private func clearSystemAudioConverter() {
        os_unfair_lock_lock(&converterLock)
        systemAudioConverter = nil
        os_unfair_lock_unlock(&converterLock)
        lastSystemPTS = .invalid
        lastSystemDuration = .invalid
    }

    // MARK: - Microphone Capture

    private func startMicrophoneCapture() throws {
        micEngine = AVAudioEngine()
        guard let engine = micEngine else { return }

        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        fputs("Mic: \(inputFormat.sampleRate)Hz, \(inputFormat.channelCount)ch\n", stderr)

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            fputs("Warning: Could not create mic converter\n", stderr)
            return
        }

        let micGain = self.micGain
        let micBuffer = self.micBuffer

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: inputFormat) { [weak self] inputBuffer, _ in
            guard let self,
                  let samples = self.convertAndExtractSamples(from: inputBuffer, using: converter, gain: micGain) else {
                return
            }
            micBuffer.write(samples)
        }

        try engine.start()
        fputs("Microphone started\n", stderr)
    }

    // MARK: - Output Loop (Timer-Based, Stereo: L=mic, R=system)

    private func startOutputLoop() {
        let systemBuffer = self.systemBuffer
        let micBuffer = self.micBuffer
        let includeMic = self.includeMic

        // Pre-allocate buffers - no allocations during timer callback
        let systemFloats = UnsafeMutablePointer<Float>.allocate(capacity: outputFrameSize)
        let micFloats = UnsafeMutablePointer<Float>.allocate(capacity: outputFrameSize)
        // Stereo output: 2 channels interleaved [L0, R0, L1, R1, ...]
        let stereoFrameCount = outputFrameSize * outputChannels
        let outputBuffer = UnsafeMutablePointer<Int16>.allocate(capacity: stereoFrameCount)
        // vDSP intermediate buffers for vectorized float-to-int16 conversion
        let interleavedFloats = UnsafeMutablePointer<Float>.allocate(capacity: stereoFrameCount)
        let clippedFloats = UnsafeMutablePointer<Float>.allocate(capacity: stereoFrameCount)
        let scaledFloats = UnsafeMutablePointer<Float>.allocate(capacity: stereoFrameCount)
        systemFloats.initialize(repeating: 0, count: outputFrameSize)
        micFloats.initialize(repeating: 0, count: outputFrameSize)
        outputBuffer.initialize(repeating: 0, count: stereoFrameCount)
        interleavedFloats.initialize(repeating: 0, count: stereoFrameCount)
        clippedFloats.initialize(repeating: 0, count: stereoFrameCount)
        scaledFloats.initialize(repeating: 0, count: stereoFrameCount)

        setvbuf(stdout, nil, _IONBF, 0)

        // 1024 samples at 16kHz = 64ms = 64,000,000 nanoseconds
        let intervalNanos = UInt64(Double(outputFrameSize) / targetSampleRate * 1_000_000_000)

        // Create high-precision timer with .strict flag to minimize coalescing
        let timer = DispatchSource.makeTimerSource(flags: .strict, queue: outputQueue)
        timer.schedule(
            deadline: .now(),
            repeating: .nanoseconds(Int(intervalNanos)),
            leeway: .nanoseconds(0)
        )

        var outputStarted = false
        var lastUnderrunReport = 0
        var lastWatchdogFrameCount: UInt64 = 0
        var watchdogStaleTicks = 0
        let watchdogThresholdTicks = 16  // ~1s at 64ms intervals
        weak let weakSelf = self

        timer.setEventHandler { [systemBuffer, micBuffer, systemFloats, micFloats, outputBuffer,
                                  interleavedFloats, clippedFloats, scaledFloats] in
            guard let strongSelf = weakSelf, strongSelf.isRunning else { return }

            let sysAvailable = systemBuffer.available

            // Pre-buffering: wait for system audio only (mic may not have input yet)
            if !outputStarted {
                if sysAvailable < minBufferSamples { return }
                outputStarted = true
                fputs("Output started (stereo: L=mic, R=system)\n", stderr)
            }

            // System audio watchdog: detect stalled stream
            let currentFrameCount = strongSelf.systemAudioFrameCount
            if currentFrameCount == lastWatchdogFrameCount {
                watchdogStaleTicks += 1
                if watchdogStaleTicks == watchdogThresholdTicks {
                    fputs("Warning: system audio stalled, attempting restart\n", stderr)
                    strongSelf.restartCapture()
                }
            } else {
                lastWatchdogFrameCount = currentFrameCount
                watchdogStaleTicks = 0
            }

            // Read from both buffers
            _ = systemBuffer.readFloat(into: systemFloats, count: outputFrameSize)
            if includeMic {
                _ = micBuffer.readFloat(into: micFloats, count: outputFrameSize)
            }

            // Interleave mic/system into float buffer
            for i in 0..<outputFrameSize {
                interleavedFloats[i * 2] = includeMic ? micFloats[i] : 0
                interleavedFloats[i * 2 + 1] = systemFloats[i]
            }

            // vDSP: clamp [-1,1], scale by 32767, convert to Int16
            var low: Float = -1.0
            var high: Float = 1.0
            vDSP_vclip(interleavedFloats, 1, &low, &high, clippedFloats, 1, vDSP_Length(stereoFrameCount))
            var scale: Float = 32767.0
            vDSP_vsmul(clippedFloats, 1, &scale, scaledFloats, 1, vDSP_Length(stereoFrameCount))
            vDSP_vfix16(scaledFloats, 1, outputBuffer, 1, vDSP_Length(stereoFrameCount))

            // Write stereo to stdout
            let bytesToWrite = stereoFrameCount * MemoryLayout<Int16>.size
            let written = fwrite(outputBuffer, 1, bytesToWrite, stdout)

            if written < bytesToWrite {
                strongSelf.outputTimer?.cancel()
                return
            }

            // Report underruns periodically
            let sysUnderruns = systemBuffer.underruns
            let micUnderruns = micBuffer.underruns
            let totalUnderruns = sysUnderruns + micUnderruns
            if totalUnderruns > lastUnderrunReport && totalUnderruns % 10 == 0 {
                fputs("Warning: underruns (sys: \(sysUnderruns), mic: \(micUnderruns))\n", stderr)
                lastUnderrunReport = totalUnderruns
            }
        }

        timer.setCancelHandler {
            systemFloats.deallocate()
            micFloats.deallocate()
            outputBuffer.deallocate()
            interleavedFloats.deallocate()
            clippedFloats.deallocate()
            scaledFloats.deallocate()
        }

        self.outputTimer = timer
        timer.resume()
    }

    // MARK: - Restart Capture

    func restartCapture() {
        Task { await performRestart() }
    }

    private func performRestart() async {
        // Atomically check-and-set restart state
        let attempt: (num: Int, max: Int)? = restartState.withLock { state in
            guard !state.isRestarting, state.count < maxRestarts else { return nil }
            // Rate limit: skip if last attempt was too recent
            if let lastAttempt = state.lastAttemptTime,
               Date().timeIntervalSince(lastAttempt) < minRestartInterval {
                return nil
            }
            state.isRestarting = true
            state.count += 1
            state.lastAttemptTime = Date()
            return (state.count, maxRestarts)
        }

        guard let attempt else {
            if restartState.withLock({ $0.count >= maxRestarts && !$0.isRestarting }) {
                fputs("Stream restart: max retries (\(maxRestarts)) exceeded\n", stderr)
            }
            return
        }
        defer { restartState.withLock { $0.isRestarting = false } }

        let backoff = min(0.25 * pow(2.0, Double(attempt.num - 1)), 10.0)  // 0.25s, 0.5s, 1s, 2s, ... capped at 10s
        fputs("Stream restart: attempt \(attempt.num)/\(attempt.max) (backoff \(Int(backoff))s)\n", stderr)
        try? await Task.sleep(nanoseconds: UInt64(backoff * 1_000_000_000))

        // Stop old stream
        try? await stream?.stopCapture()
        stream = nil

        clearSystemAudioConverter()

        // Re-query content and create fresh stream
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
            guard let display = content.displays.first else {
                throw CaptureError.noDisplay
            }
            let config = makeStreamConfig()
            let filter = SCContentFilter(display: display, excludingWindows: [])

            outputHandler = StreamOutputHandler(manager: self)
            stream = SCStream(filter: filter, configuration: config, delegate: outputHandler)
            try stream?.addStreamOutput(outputHandler!, type: .audio, sampleHandlerQueue: audioSampleQueue)
            try stream?.addStreamOutput(outputHandler!, type: .screen, sampleHandlerQueue: screenDropQueue)

            try await stream?.startCapture()
            restartState.withLock { $0.lastTime = Date() }
            fputs("Stream restart: success\n", stderr)
        } catch {
            fputs("Stream restart: failed - \(error.localizedDescription)\n", stderr)
        }
    }

    // MARK: - Stop

    func stop() async {
        isRunning = false
        outputTimer?.cancel()
        outputTimer = nil
        micEngine?.inputNode.removeTap(onBus: 0)
        micEngine?.stop()
        try? await stream?.stopCapture()
        stream = nil
    }
}

// MARK: - Stream Output Handler

final class StreamOutputHandler: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    weak var manager: AudioCaptureManager?

    init(manager: AudioCaptureManager) {
        self.manager = manager
        super.init()
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio else { return }
        manager?.processSystemAudio(sampleBuffer)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        let nsError = error as NSError
        fputs("Stream error: [\(nsError.domain) \(nsError.code)] \(nsError.localizedDescription)\n", stderr)
        // -3808 = attemptToStopStreamState (intentional stop), don't restart
        if nsError.code == -3808 { return }
        manager?.restartCapture()
    }
}

// MARK: - Errors

enum CaptureError: Error, LocalizedError {
    case noDisplay
    var errorDescription: String? { "No display found" }
}

// MARK: - Main

let args = CommandLine.arguments
let noMic = args.contains("--no-mic")
let listDevicesFlag = args.contains("--list")
let help = args.contains("--help") || args.contains("-h")

if help {
    fputs("""
    audiocapture - Capture system audio and microphone

    Usage: audiocapture [options]

    Options:
      --no-mic    Disable microphone capture (right channel only)
      --list      List available audio devices
      --help, -h  Show this help

    Output: 16-bit signed PCM, 16kHz, stereo (L=mic, R=system)

    The stereo output allows speaker detection:
      - Left channel: microphone (you)
      - Right channel: system audio (other speaker)

    Example:
      audiocapture | ffplay -f s16le -ar 16000 -ch_layout stereo -

    """, stderr)
    exit(0)
}

let manager = AudioCaptureManager()

if listDevicesFlag {
    manager.listDevices()
    exit(0)
}

manager.includeMic = !noMic

signal(SIGINT) { _ in
    fputs("\nStopping...\n", stderr)
    exit(0)
}

signal(SIGPIPE) { _ in
    exit(0)
}

Task {
    do {
        try await manager.startCapture()
        while true { try await Task.sleep(nanoseconds: 1_000_000_000) }
    } catch {
        fputs("Error: \(error.localizedDescription)\n", stderr)
        exit(1)
    }
}

RunLoop.main.run()
