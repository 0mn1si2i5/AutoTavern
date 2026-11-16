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
                    "running_game_pids": apps.map { $0.processIdentifier },
                    "frontmost_pid": NSWorkspace.shared.frontmostApplication?.processIdentifier ?? -1, "windows": []]
                if granted {
                    do { result["windows"] = try await windows().map(windowInfo) }
                    catch { result["window_enumeration_error"] = String(describing: error) }
                }
                try emit(result)
            } else if command == "scan" {
                guard args.count == 2,
                      let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: args[1]) as CFURL, nil),
                      let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw ProbeError("scan_image_required") }
                let w = img.width, h = img.height
                var bytes = [UInt8](repeating: 0, count: w*h*4)
                let result = bytes.withUnsafeMutableBytes { raw -> [String: Any] in
                    let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8,
                                        bytesPerRow: w*4, space: CGColorSpaceCreateDeviceRGB(),
                                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
                    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
                    let b = raw.bindMemory(to: UInt8.self)
                    var bands: [String: [Int]] = [:]
                    for (name, lo, hi) in [("shop", 0.42, 0.46), ("board", 0.59, 0.63), ("hand", 0.88, 0.965)] {
                        var columns = [Int](repeating: 0, count: w)
                        for y in Int(lo*Double(h))..<Int(hi*Double(h)) {
                            for x in Int(0.22*Double(w))..<Int(0.72*Double(w)) {
                                let i = (y*w+x)*4
                                let r = Int(b[i]), g = Int(b[i+1]), blue = Int(b[i+2])
                                let hit = name == "hand" ? (g > 180 && r < 110 && blue < 150) :
                                    (r > 145 && r > g*2 && r > blue*2)
                                if hit { columns[x] += 1 }
                            }
                        }
                        bands[name] = columns
                    }
                    var upgradeGreenPixels = 0
                    for y in Int(0.18*Double(h))..<Int(0.27*Double(h)) {
                        for x in Int(0.37*Double(w))..<Int(0.435*Double(w)) {
                            let i = (y*w+x)*4
                            if b[i+1] > 180 && b[i] < 110 && b[i+2] < 150 { upgradeGreenPixels += 1 }
                        }
                    }
                    return ["width": w, "height": h, "bands": bands,
                            "upgrade_green_pixels": upgradeGreenPixels]
                }
                try emit(result)
            } else if command == "ocr" {
                guard args.count == 2 || args.count == 6 else { throw ProbeError("ocr requires image and optional normalized x y width height") }
                let url = URL(fileURLWithPath: args[1])
                guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                      let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
                    throw ProbeError("image_decode_failed")
                }
                guard image.width <= 16384 && image.height <= 16384 else { throw ProbeError("image_too_large") }
                var region = CGRect(x: 0, y: 0, width: 1, height: 1)
                if args.count == 6 {
                    let values = args[2...5].compactMap(Double.init)
                    guard values.count == 4, values.allSatisfy({ $0.isFinite }),
                          values[0] >= 0, values[1] >= 0, values[2] > 0, values[3] > 0,
                          values[0]+values[2] <= 1, values[1]+values[3] <= 1 else {
                        throw ProbeError("invalid_ocr_region")
                    }
                    region = CGRect(x: values[0], y: values[1], width: values[2], height: values[3])
                }
                let pixelRegion = CGRect(x: region.minX*Double(image.width), y: region.minY*Double(image.height),
                                         width: region.width*Double(image.width), height: region.height*Double(image.height)).integral
                guard let cropped = image.cropping(to: pixelRegion) else { throw ProbeError("crop_failed") }
                let rows = try textRows(cropped).map { item -> [String: Any] in
                    var mapped = item
                    let box = item["box"] as! [String: CGFloat]
                    mapped["box"] = ["x": (pixelRegion.minX+box["x"]!*pixelRegion.width)/Double(image.width),
                                     "y": (pixelRegion.minY+box["y"]!*pixelRegion.height)/Double(image.height),
                                     "width": box["width"]!*pixelRegion.width/Double(image.width),
                                     "height": box["height"]!*pixelRegion.height/Double(image.height)]
                    return mapped
                }
                try emit(["width": image.width, "height": image.height, "text": rows,
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
