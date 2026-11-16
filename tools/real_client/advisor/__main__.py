"""OmniBob: read-only state, hotkey advice, human execution."""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
import signal
import sqlite3
import subprocess
import time
from urllib.parse import urlparse

from .runtime import Advisor, ROOT, atomic, now
from .source import LogSource, ReplaySource


class Catalog(dict):
    def __init__(self, path):
        self.path = Path(path).resolve()
        db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        self.meta = dict(db.execute("SELECT key,value FROM metadata"))
        cards = [json.loads(row[0]) for row in db.execute("SELECT data FROM cards")]
        db.close()
        super().__init__((card["id"], card) for card in cards)
        self.by_dbf = {card["dbfId"]: card for card in cards}

    def get(self, key, default=None):
        return self.by_dbf.get(key, default) if isinstance(key, int) else super().get(key, default)


def catalog_path(path):
    if path:
        return path
    pointer = ROOT / "runs/real-client-data/CURRENT.json"
    if not pointer.exists():
        raise FileNotFoundError("没有本地卡表，请用 --catalog 指定已有 catalog.sqlite")
    return Path(json.loads(pointer.read_text())["catalog"])


def handler_for(advisor, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # No cookies, URL tokens, or request data in terminal logs.

        def reply(self, value, code=200):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            supplied = self.headers.get("X-Omnibob-Token", "")
            return secrets.compare_digest(supplied, token)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                data = Path(__file__).with_name("panel.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/state" and self.authorized():
                try:
                    advisor.observe()
                except (OSError, ValueError, KeyError):
                    pass  # observe has already invalidated and recorded the state.
                self.reply(advisor.state)
            else:
                self.reply({"error": "Not found or missing session token"}, 404)

        def do_POST(self):
            if not self.authorized():
                self.reply({"error": "Invalid session token"}, 403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 <= size <= 16384:
                    raise ValueError("Request too large")
                body = json.loads(self.rfile.read(size) or b"{}")
                path = urlparse(self.path).path
                if path == "/api/refresh":
                    result = advisor.refresh()
                elif path == "/api/replay/next":
                    result = advisor.navigate(1)
                elif path == "/api/replay/previous":
                    result = advisor.navigate(-1)
                elif path == "/api/note":
                    result = advisor.note(body.get("text"))
                else:
                    self.reply({"error": "Unknown command"}, 404)
                    return
                self.reply(result)
            except (ValueError, OSError, TypeError, AttributeError) as error:
                self.reply({"error": str(error)}, 400)

    return Handler


def native_panel():
    source = Path(__file__).with_name("AdvisorPanel.swift")
    version = subprocess.check_output(["xcrun", "swiftc", "--version"], text=True, timeout=15)
    digest = hashlib.sha256(source.read_bytes() + version.encode()).hexdigest()[:20]
    out = ROOT / "runs/advisor-tools" / digest
    out.mkdir(parents=True, exist_ok=True)
    binary = out / "AdvisorPanel"
    receipt = out / "BUILD.json"
    if binary.exists() and receipt.exists():
        if hashlib.sha256(binary.read_bytes()).hexdigest() == json.loads(receipt.read_text())["sha256"]:
            return binary
        raise ValueError("面板缓存校验不符，请保留该目录检查")
    command = ["xcrun", "swiftc", str(source), "-o", str(binary), "-framework", "AppKit",
               "-framework", "Carbon", "-framework", "WebKit",
               "-module-cache-path", str(ROOT / "runs/advisor-tools/module-cache")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (out / "build.stdout").write_text(result.stdout)
    (out / "build.stderr").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"面板编译失败，见 {out / 'build.stderr'}")
    atomic(receipt, {"command": command, "compiler": version,
                    "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()})
    return binary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("live", "replay"))
    parser.add_argument("--log", type=Path, help="Power.log; live 未指定时自动发现")
    parser.add_argument("--log-directory", type=Path)
    parser.add_argument("--catalog", type=Path, help="本地只读 SQLite 卡表")
    parser.add_argument("--release", choices=("v0.2", "v0.3"), help="不修改默认指针")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--panel", action="store_true", help="显示原生浮动面板及全局刷新快捷键")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--start", type=int, default=1, help="回放开始行，1基")
    parser.add_argument("--export", action="store_true", help="离线逐操作建议导出后退出；不打模拟对局")
    args = parser.parse_args(argv)
    if args.mode == "replay" and not args.log:
        parser.error("replay requires --log")
    if args.export and args.mode != "replay":
        parser.error("--export is only available with replay")
    out = args.output or ROOT / "runs/advisor" / (args.mode + "-" + now().replace(":", "").replace("+", "_"))
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    catalog = Catalog(catalog_path(args.catalog))
    source = ReplaySource(catalog, args.log) if args.mode == "replay" else LogSource(catalog, args.log, args.log_directory)
    advisor = Advisor(source, catalog, out, args.release)
    if args.mode == "replay" and not source.goto(args.start - 1):
        advisor.close()
        parser.error("--start is outside the replay")
    server, panel = None, None
    try:
        if args.export:
            counts = {}
            for index in range(args.start - 1, source.total):
                if stopped or (out / "STOP").exists():
                    break
                source.goto(index)
                result = advisor.refresh()
                counts[result["status"]] = counts.get(result["status"], 0) + 1
            atomic(out / "REPLAY_RESULT.json", {"source": source.info(), "frames": sum(counts.values()),
                   "counts": counts, "source_unchanged": source._digest == hashlib.sha256(source.path.read_bytes()).hexdigest(),
                   "scope": "fixed recorded states; no simulated games or strength evaluation",
                   "complete": sum(counts.values()) == source.total - args.start + 1})
            print(json.dumps({"output": str(out), "frames": sum(counts.values()), "counts": counts}, ensure_ascii=False))
            return
        token = secrets.token_urlsafe(24)
        server = HTTPServer(("127.0.0.1", args.port), handler_for(advisor, token))
        server.timeout = .2
        url = f"http://127.0.0.1:{server.server_port}/#token={token}"
        atomic(out / "SERVER.json", {"url": url, "pid": __import__("os").getpid()})
        print(f"OmniBob 只读建议面板：{url}\n记录：{out}\n关闭面板或 Ctrl-C 停止。", flush=True)
        if args.mode == "replay":
            advisor.refresh()
        if args.panel:
            binary = native_panel()
            panel = subprocess.Popen([str(binary), "--url", url])
        last_poll = 0
        while not stopped and not (out / "STOP").exists():
            if panel is not None and panel.poll() is not None:
                break
            server.handle_request()
            if time.monotonic() - last_poll >= .5:
                try:
                    advisor.observe()
                except (OSError, ValueError, KeyError):
                    pass
                last_poll = time.monotonic()
    finally:
        if server is not None:
            server.server_close()
        if panel is not None and panel.poll() is None:
            panel.terminate()  # This child is only our own display, never the client.
            panel.wait(timeout=5)
        advisor.close()


if __name__ == "__main__":
    main()
