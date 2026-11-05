import Foundation
import AppKit
import CoreGraphics
import ScreenCaptureKit
import Vision
import CoreML
import ImageIO
import UniformTypeIdentifiers
import ApplicationServices

// Read-only probe: no mouse/keyboard generation, process memory, or network APIs.
let hearthstoneBundle = "unity.Blizzard Entertainment.Hearthstone"

struct ProbeError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

func emit(_ object: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    print(String(decoding: data, as: UTF8.self))
}

func textRows(_ image: CGImage) throws -> [[String: Any]] {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    // The current client is Chinese. English-first recognition misreads Chinese
    // UI labels as Latin strings even when Chinese is listed as a fallback.
    request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en-US"]
    request.automaticallyDetectsLanguage = true
    let supported = try request.supportedRecognitionLanguages()
    request.recognitionLanguages = request.recognitionLanguages.filter { supported.contains($0) }
    // Explicit CPU request; initial model compilation can still be slow on this
    // OS. Device selection is configuration, not a measured hardware-use claim.
    for (stage, devices) in try request.supportedComputeStageDevices {
        guard let cpu = devices.first(where: { if case .cpu = $0 { return true }; return false }) else {
            throw ProbeError("ocr_cpu_device_unavailable")
        }
        request.setComputeDevice(cpu, for: stage)
    }
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    return (request.results ?? []).compactMap { item in
        guard let candidate = item.topCandidates(1).first else { return nil }
        let box = item.boundingBox
        // Vision is bottom-left; archive coordinates are top-left, normalized to PNG.
        return ["text": candidate.string, "confidence": candidate.confidence,
                "box": ["x": box.minX, "y": 1-box.maxY, "width": box.width, "height": box.height]]
    }
}

func windows() async throws -> [SCWindow] {
    guard CGPreflightScreenCaptureAccess() else {
        throw ProbeError("screen_recording_permission_required")
    }
    let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
    return content.windows.filter {
        $0.owningApplication?.bundleIdentifier == hearthstoneBundle && $0.windowLayer == 0 &&
            $0.frame.width > 0 && $0.frame.height > 0
    }
}

func windowInfo(_ w: SCWindow) -> [String: Any] {
    return ["window_id": w.windowID, "pid": w.owningApplication?.processID ?? 0,
            "title": w.title ?? "", "bundle_id": w.owningApplication?.bundleIdentifier ?? "",
            "frame_points": ["x": w.frame.minX, "y": w.frame.minY,
                             "width": w.frame.width, "height": w.frame.height]]
}

@main
struct ScreenProbe {
    @MainActor
    static func main() async {
        // Initialize AppKit's WindowServer connection on the main actor before
        // async ScreenCaptureKit calls. This creates no window or input event.
        _ = NSApplication.shared
        do {
            let args = Array(CommandLine.arguments.dropFirst())
            guard let command = args.first else { throw ProbeError("expected doctor|capture WINDOW_ID OUTPUT.png|ocr INPUT.png") }
            if command == "doctor" {
                let granted = CGPreflightScreenCaptureAccess()
                let apps = NSRunningApplication.runningApplications(withBundleIdentifier: hearthstoneBundle)
                var result: [String: Any] = ["screen_recording_permission": granted,
                    "accessibility_permission": AXIsProcessTrusted(), "readonly": true,
                    "running_game_pids": apps.map { $0.processIdentifier }, "windows": []]
                if granted {
                    do { result["windows"] = try await windows().map(windowInfo) }
                    catch { result["window_enumeration_error"] = String(describing: error) }
                }
                try emit(result)
            } else if command == "ocr" {
                guard args.count == 2 else { throw ProbeError("ocr requires one image") }
                let url = URL(fileURLWithPath: args[1])
                guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                      let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                    throw ProbeError("image_decode_failed")
                }
                guard image.width <= 16384 && image.height <= 16384 else { throw ProbeError("image_too_large") }
                try emit(["width": image.width, "height": image.height, "text": try textRows(image),
                          "ocr_only": true, "ocr_compute_requested": "cpu", "coordinate_system": "normalized_top_left"])
            } else if command == "capture" {
                guard args.count == 3, let id = UInt32(args[1]) else { throw ProbeError("capture requires WINDOW_ID OUTPUT.png") }
                let available = try await windows()
                guard let window = available.first(where: { $0.windowID == id }) else {
                    throw ProbeError("hearthstone_window_not_visible_or_changed")
                }
                let path = URL(fileURLWithPath: args[2])
                guard !FileManager.default.fileExists(atPath: path.path) else { throw ProbeError("refuse_image_overwrite") }
                let filter = SCContentFilter(desktopIndependentWindow: window)
                let config = SCStreamConfiguration()
                config.width = max(1, Int(filter.contentRect.width * CGFloat(filter.pointPixelScale)))
                config.height = max(1, Int(filter.contentRect.height * CGFloat(filter.pointPixelScale)))
                config.showsCursor = false
                config.ignoreShadowsSingleWindow = true
                let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
                let data = NSMutableData()
                guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
                    throw ProbeError("png_encoder_unavailable")
                }
                CGImageDestinationAddImage(destination, image, nil)
                guard CGImageDestinationFinalize(destination) else { throw ProbeError("png_encode_failed") }
                try (data as Data).write(to: path, options: .withoutOverwriting)
                try emit(["window": windowInfo(window), "width": image.width, "height": image.height,
                          "image": path.path, "text": [], "readonly": true,
                          "ocr_status": "not_run", "coordinate_system": "normalized_top_left"])
            } else { throw ProbeError("unknown_command") }
        } catch {
            try? emit(["error": String(describing: error), "readonly": true])
            exit(2)
        }
    }
}
