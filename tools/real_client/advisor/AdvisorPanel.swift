// A read-only companion window. Registers a shortcut for this app only;
// there is no event injection, client activation, or accessibility automation.
import AppKit
import Carbon
import WebKit

final class AdvicePanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}

final class Delegate: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate {
    let url: URL
    var panel: AdvicePanel!
    var web: WKWebView!
    var hotkey: EventHotKeyRef?
    var eventHandler: EventHandlerRef?
    var shortcutMessage = ""

    init(url: URL) { self.url = url }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        web = WKWebView(frame: .zero, configuration: config)
        web.navigationDelegate = self
        let available = NSScreen.main?.visibleFrame ?? NSRect(x: 50, y: 50, width: 1280, height: 900)
        let width = min(600, available.width * 0.45)
        let height = min(850, available.height - 48)
        panel = AdvicePanel(
            contentRect: NSRect(x: available.maxX - width - 18, y: available.maxY - height - 24,
                               width: width, height: height),
            styleMask: [.titled, .closable, .resizable, .nonactivatingPanel],
            backing: .buffered, defer: false)
        panel.title = "OmniBob · 只读建议 · ⌃⌥Space 刷新"
        panel.minSize = NSSize(width: 370, height: 430)
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.hidesOnDeactivate = false
        panel.becomesKeyOnlyIfNeeded = true
        panel.isReleasedWhenClosed = false
        panel.delegate = self
        panel.contentView = web
        registerShortcut()
        web.load(URLRequest(url: url))
        // Show our panel without activating Hearthstone or taking keyboard focus.
        panel.orderFrontRegardless()
    }

    func registerShortcut() {
        var event = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        let context = Unmanaged.passUnretained(self).toOpaque()
        let installed = InstallEventHandler(GetApplicationEventTarget(), { _, event, context in
            guard let context, let event else { return OSStatus(eventNotHandledErr) }
            var identifier = EventHotKeyID()
            let read = GetEventParameter(event, EventParamName(kEventParamDirectObject),
                                        EventParamType(typeEventHotKeyID), nil,
                                        MemoryLayout<EventHotKeyID>.size, nil, &identifier)
            guard read == noErr, identifier.signature == 0x4F4D4E49, identifier.id == 1 else {
                return OSStatus(eventNotHandledErr)
            }
            let app = Unmanaged<Delegate>.fromOpaque(context).takeUnretainedValue()
            app.web.evaluateJavaScript("window.omniRefresh && window.omniRefresh(); void 0", completionHandler: nil)
            return noErr
        }, 1, &event, context, &eventHandler)
        guard installed == noErr else {
            shortcutMessage = "全局快捷键未注册（\(installed)）；可点击刷新，或在面板内按 ⌘Enter。"
            return
        }
        let key = EventHotKeyID(signature: 0x4F4D4E49, id: 1)
        let registered = RegisterEventHotKey(UInt32(kVK_Space), UInt32(controlKey | optionKey), key,
                                             GetApplicationEventTarget(), 0, &hotkey)
        if registered != noErr {
            shortcutMessage = "⌃⌥Space 可能被其他应用占用（\(registered)）；可点击刷新，或在面板内按 ⌘Enter。"
        }
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        if let bytes = try? JSONSerialization.data(withJSONObject: [shortcutMessage]),
           let literal = String(data: bytes, encoding: .utf8) {
            web.evaluateJavaScript("window.setNativeHotkeyStatus((\(literal))[0]);", completionHandler: nil)
        }
    }

    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let target = navigationAction.request.url,
              target.scheme == "http", target.host == url.host, target.port == url.port else {
            decisionHandler(.cancel)
            return
        }
        decisionHandler(.allow)
    }

    func windowWillClose(_ notification: Notification) { NSApp.terminate(nil) }

    func applicationWillTerminate(_ notification: Notification) {
        if let hotkey { UnregisterEventHotKey(hotkey) }
        if let eventHandler { RemoveEventHandler(eventHandler) }
    }
}

let arguments = CommandLine.arguments
guard let flag = arguments.firstIndex(of: "--url"), arguments.indices.contains(flag + 1),
      let url = URL(string: arguments[flag + 1]), url.scheme == "http",
      url.host == "127.0.0.1", let port = url.port, (1...65535).contains(port),
      url.user == nil, url.password == nil else {
    fputs("Usage: AdvisorPanel --url 'http://127.0.0.1:PORT/#token=TOKEN'\n", stderr)
    exit(2)
}
let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let delegate = Delegate(url: url)
app.delegate = delegate
app.run()
