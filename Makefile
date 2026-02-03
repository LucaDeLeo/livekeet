.PHONY: build clean test install

# Build the Swift audio capture tool
build:
	cd audiocapture && swift build -c release

# Clean build artifacts
clean:
	cd audiocapture && swift package clean

# Test audio capture (plays through speakers) - stereo: L=mic, R=system
test-audio:
	./audiocapture/.build/release/audiocapture | ffplay -f s16le -ar 16000 -ac 2 -nodisp -autoexit -

# Test system audio only (right channel only)
test-system:
	./audiocapture/.build/release/audiocapture --no-mic | ffplay -f s16le -ar 16000 -ac 2 -nodisp -autoexit -

# List audio devices
list-devices:
	./audiocapture/.build/release/audiocapture --list

# Install Python package and build audio capture
install: build
	uv sync
	@echo ""
	@echo "Done! Run 'livekeet' to start transcribing."
