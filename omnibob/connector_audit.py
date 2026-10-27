"""Local logical-request journal around the unchanged reviewed arena connector.

The official connector still owns authentication, retries, queueing and timing.
This wrapper never logs headers, keys, arbitrary error messages or URL queries.
It records logical requests, not each internal HTTP retry. Running a real match
still requires the user's authorization and locally supplied ARENA_API_KEY.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

REVIEWED_CONNECTOR_SHA256 = "c23376504d809be575768e4453b4228c76dc274ab628799d4985b56a2b7b5622"
REQUEST_FIELDS = {"match_id", "seat", "seat_revision", "decision_id", "action", "request_id",
                  "mode", "ruleset_id", "wait_for_human", "min_players", "preferred_players",
                  "max_wait_seconds", "timeout_action", "model_version", "games"}
RESPONSE_FIELDS = {"type", "match_id", "seat", "seat_revision", "decision_id", "turn", "phase",
                   "remaining_ms", "actions_used", "action_limit", "player_view", "observation",
                   "status", "ok", "accepted", "action", "placement", "placements", "client_error",
                   "ruleset_id", "protocol_version", "catalog_sha256", "observation_sha256",
                   "action_layout_sha256"}
ERROR_CODES = {"stale", "expired", "illegal", "forbidden", "match_void", "action_limit",
               "already_playing", "unauthorized", "rate_limited"}


def selected(value, fields):
    return {key: value[key] for key in fields if key in value} if isinstance(value, dict) else {}


def instrument(module, output):
    """Attach logging to an already loaded connector; useful for loopback integration."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    lock = threading.Lock()
    sequence = 0
    original = module.ArenaClient

    def emit(event):
        nonlocal sequence
        with lock:
            sequence += 1
            event = {"sequence": sequence, "time_unix": time.time(), **event}
            with (output/"requests.jsonl").open("a") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+"\n")
            return sequence

    class AuditedClient(original):
        def request(self, method, path, body=None, authenticated=True):
            route = path.split("?", 1)[0]
            operation = emit({"kind": "request", "method": method, "route": route,
                              "body": selected(body, REQUEST_FIELDS)})
            start = time.perf_counter()
            try:
                response = super().request(method, path, body, authenticated)
            except BaseException as error:
                emit({"kind": "exception", "operation": operation,
                      "exception_type": type(error).__name__, "elapsed_seconds": time.perf_counter()-start})
                raise
            safe = selected(response, RESPONSE_FIELDS)
            if isinstance(response, dict) and "error" in response:
                code = response["error"]
                safe["error"] = code if isinstance(code, str) and code in ERROR_CODES else "unrecognized_error"
            emit({"kind": "response", "operation": operation, "elapsed_seconds": time.perf_counter()-start,
                  "response": safe})
            return response

    module.ArenaClient = AuditedClient
    (output/"scope.json").write_text(json.dumps({
        "scope": "logical requests and their final responses; internal wire retries are not individually logged",
        "visibility": "own normal decision observations and final results; never headers or Agent Key",
        "limitations": "A successful logical response is server-reported evidence, not independent proof of game correctness. "
                       "Process restart creates a new journal; a partial operation remains incomplete."}, indent=2)+"\n")
    return module


def load_reviewed_connector(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != REVIEWED_CONNECTOR_SHA256:
        raise ValueError("connector checksum differs from the reviewed version; review the new client explicitly")
    spec = importlib.util.spec_from_file_location("omnibob_reviewed_connector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new local audit directory")
    parser.add_argument("connector_args", nargs=argparse.REMAINDER, help="official arguments after --; use --check for anonymous validation")
    args = parser.parse_args()
    module = instrument(load_reviewed_connector(args.client), args.output)
    from .compat import digest, environment_contract, source_manifest
    checkpoint = os.environ.get("OMNIBOB_CHECKPOINT")
    metadata = {"connector_sha256": REVIEWED_CONNECTOR_SHA256,
                "source_sha256": digest(source_manifest()), "environment": environment_contract(),
                "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None}
    (args.output/"provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
    forwarded = args.connector_args[1:] if args.connector_args[:1] == ["--"] else args.connector_args
    previous = sys.argv
    try:
        sys.argv = [str(args.client), *forwarded]
        return module.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
