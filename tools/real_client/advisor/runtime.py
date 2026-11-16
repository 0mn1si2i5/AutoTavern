"""Read-only advisory sessions. Model proposals never dispatch game input."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .state import public_view
from .encoding import Encoder

ROOT = Path(__file__).resolve().parents[3]


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def fingerprint(view):
    return hashlib.sha256(json.dumps(view, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def proposal_runtime(policy):
    """Copy controller state, share immutable inference weights.

    Generic deepcopy recurses through the wrappers' delegated __getattr__ before
    their base exists. Construct the two supported wrappers explicitly instead.
    """
    from omnibob.repeat_guard import RepeatGuardPolicy
    from omnibob.move_guard import MoveGuardPolicy
    if isinstance(policy, RepeatGuardPolicy):
        result = RepeatGuardPolicy(proposal_runtime(policy.base), policy.repeat_limit)
        result.turn, result.counts = policy.turn, policy.counts.copy()
        return result
    if isinstance(policy, MoveGuardPolicy):
        result = MoveGuardPolicy(proposal_runtime(policy.base), policy.move_guard_budget)
        result.turn, result.context, result.moves = policy.turn, policy.context, policy.moves
        result.seen = policy.seen.copy()
        return result
    return policy


class Advisor:
    def __init__(self, source, catalog, session, version=None):
        self.source, self.catalog = source, catalog
        self.session = Path(session).resolve()
        self.session.mkdir(parents=True, exist_ok=False)
        self.mode = "replay" if hasattr(source, "goto") else "live"
        self.encoder = Encoder(catalog)
        self.policy, self.release, self.version = None, {}, version
        self.advice_key = None
        self.policy_context = None
        self.cache = {}
        self.cached_runtimes = {}
        self.pending = None
        self.actual_seen = {}
        self.actual_generation = None
        self.sequence = 0
        self.started = now()
        self.state = {"mode": self.mode, "status": "waiting", "message": "等待可见局面",
                      "view": {}, "advice": None, "warnings": [], "model": {},
                      "source": source.info(), "session_path": str(self.session),
                      "updated_at": now(), "hotkey": "Ctrl + Option + Space"}
        catalog_path = getattr(catalog, "path", None)
        source_paths = list(Path(__file__).parent.glob("*.py"))
        source_paths += [Path(__file__).with_name("panel.html"), Path(__file__).with_name("AdvisorPanel.swift")]
        source_paths += list((Path(__file__).parents[1] / "offline_replay").glob("*.py"))
        atomic(self.session / "SESSION.json", {
            "created_at": self.started, "mode": self.mode, "source": source.info(),
            "requested_release": version or "local default", "catalog": catalog.meta,
            "catalog_path": str(catalog_path) if catalog_path else None,
            "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest() if catalog_path else None,
            "scope": "read visible state; local frozen inference; human execution",
            "training": False, "game_inputs": False,
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in source_paths}})

    def event(self, kind, **data):
        with (self.session / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"utc": now(), "kind": kind, **data},
                                    ensure_ascii=False, allow_nan=False) + "\n")

    def save(self):
        self.state["updated_at"] = now()
        atomic(self.session / "STATE.json", self.state)

    def observe(self):
        try:
            changed = self.source.poll()
            self.sync_actual()
            view = public_view(self.source.replay, self.catalog)
            info = self.source.info()
            if info.get("available") is False:
                view.update(ready=False, message=info.get("reason"), warnings=[info.get("reason")])
            view["game_key"] = str(self.source.info().get("generation", 0)) + ":" + str(view.get("game_key"))
            key = fingerprint(view)
            if self.state["view"] != view:
                self.state.update(view=view, warnings=list(view.get("warnings", [])),
                                  source=self.source.info())
                if key != self.advice_key:
                    self.state.update(advice=None, status="stale" if self.advice_key else "waiting",
                                      message="局面已更新，按快捷键获取建议")
                    if view.get("game_complete"):
                        self.state.update(status="complete", message="本局已结束，记录已保留")
                    elif info.get("available") is False:
                        self.state.update(status="waiting", message=info.get("reason"))
                self.save()
            elif changed:
                self.state["source"] = self.source.info()
            return view, key
        except (ValueError, OSError, KeyError) as error:
            self.state.update(status="error", advice=None, message=f"状态读取失败：{error}")
            self.save()
            raise

    @staticmethod
    def matches(binding, row):
        """Compare a proposal with a recorded client send, not with a keystroke."""
        actual = row.get("binding", {})
        entity = binding.get("entity") or {}
        eid = entity.get("entity_id")
        kind = binding.get("kind")
        if kind == "choose":
            return actual.get("chosen") == [eid]
        if kind == "magnetize":
            target = (binding.get("target") or {}).get("entity_id")
            attached = row.get("execution", {}).get("attached_to")
            if attached is None:
                return None  # An ordinary play send alone does not prove magnetism.
            return actual.get("source") == eid and attached == target
        sequence = row.get("resolved_magnetic", row.get("mapping", {})).get("native_action_sequence", [])
        if not sequence:
            return None
        if sequence[0] != binding.get("native_action"):
            return False
        if actual.get("option") != binding.get("option_index"):
            # Simple buttons have stable native action codes but no source slot.
            if not sequence or sequence[0] != binding.get("native_action"):
                return False
        if kind in ("buy", "sell") and actual.get("target") != eid:
            return False
        if kind in ("play", "magnetize", "move", "activate", "power") and actual.get("source") != eid:
            return False
        if kind in ("play", "move") and entity.get("card_type") == "MINION":
            if actual.get("position") != binding.get("position", 0) + 1:
                return False
        target = binding.get("selected_target") or binding.get("target")
        if isinstance(target, dict) and actual.get("target") != target.get("entity_id"):
            return False
        if "selected_suboption" in binding and actual.get("sub") != binding["selected_suboption"]:
            return False
        return True

    def sync_actual(self):
        """Append observed sends and receipt updates; never infer an action from refresh."""
        generation = self.source.info().get("generation", 0)
        if self.actual_generation != generation:
            self.actual_seen.clear()
            self.pending = None
            self.actual_generation = generation
        rows = self.source.replay.rows if self.source.replay is not None else []
        # Previously finalized rows do not change; at most recent sends await a receipt.
        start = max(0, len(self.actual_seen) - 2)
        for index, row in enumerate(rows[start:], start):
            value = {"generation": generation, "row": index, "epoch": row.get("epoch"),
                     "offset": row.get("offset"), "round": row.get("round"),
                     "event": row.get("event"), "binding": row.get("binding"),
                     "mapping": row.get("mapping"), "execution": row.get("execution")}
            key = (generation, index)
            digest = fingerprint(value)
            if self.actual_seen.get(key) != digest:
                self.actual_seen[key] = digest
                self.event("observed_client_action", **value)
        pending = self.pending
        if pending and pending["generation"] == generation and len(rows) > pending["row"]:
            row = rows[pending["row"]]
            status = row.get("execution", {}).get("status")
            if status == "confirmed":
                context_matches = all(row.get(k) == v for k, v in pending.get("context", {}).items())
                matched = self.matches(pending["binding"], row) if context_matches else False
                if matched:
                    self.policy = pending["policy"]
                self.event("proposal_receipt", decision=pending["sequence"], matched=matched,
                           actual_offset=row.get("offset"), status=status)
                self.pending = None
            elif len(rows) > pending["row"] + 1:
                self.event("proposal_receipt", decision=pending["sequence"], matched=None,
                           actual_offset=row.get("offset"), status=status)
                self.pending = None

    def _load_model(self):
        if self.policy is not None:
            return
        from omnibob.deployment import resolve
        from omnibob.learned import load_policy
        self.release = resolve(self.version)
        self.policy = load_policy(self.release["checkpoint"], device="cpu", cpu_threads=1)
        self.state["model"] = self.release
        atomic(self.session / "MODEL.json", self.release)

    def refresh(self):
        self.sequence += 1
        folder = self.session / "decisions" / f"{self.sequence:06d}"
        folder.mkdir(parents=True)
        try:
            view, key = self.observe()
            atomic(folder / "VISIBLE.json", {"source": self.source.info(), "view": view})
            if not view.get("ready"):
                self.advice_key = None
                self.state.update(status="complete" if view.get("game_complete") else "waiting", advice=None,
                                  message=view.get("reason") or view.get("message") or "等待经营阶段或完整展示的选择")
            elif key in self.cache:
                cached = self.cache[key]
                self.advice_key = key
                self.state.update(status="ready", message="局面未变，保留同一建议", **cached)
                self.set_pending(view, cached["advice"]["binding"], self.cached_runtimes[key])
                atomic(folder / "CACHED.json", {"state_sha256": key})
            else:
                obs, report = self.encoder.encode(view)
                atomic(folder / "ENCODING.json", report)
                np.savez_compressed(folder / "INPUT.npz", features=obs.features, cards=obs.cards,
                                    action_mask=obs.action_mask)
                self._load_model()
                context = (view.get("game_key"), view.get("round"))
                if context != self.policy_context:
                    if hasattr(self.policy, "reset"):
                        self.policy.reset()
                    self.policy_context = context
                    self.pending = None
                start = time.perf_counter()
                # Proposal generation must not count as an executed operation in
                # frozen runtime guards. Commit only after a matching receipt.
                proposed_policy = proposal_runtime(self.policy)
                decision = proposed_policy.act(obs)
                binding = report["bindings"][str(decision.action)]
                binding = self.encoder.complete_advice(proposed_policy, view, report, decision.action)
                binding["native_action"] = decision.action
                advice = self.describe(binding, decision.action)
                advice["inference_ms"] = round(1000 * (time.perf_counter() - start), 3)
                advice["model_reason"] = decision.reason
                advice["state_sha256"] = key
                _, fresh_key = self.observe()
                if fresh_key != key:
                    atomic(folder / "DISCARDED.json", {"reason": "state changed during inference",
                           "computed_advice": advice, "old_state_sha256": key, "new_state_sha256": fresh_key})
                    self.state.update(status="stale", advice=None,
                                      message="推理期间局面已变化，请再次刷新")
                    self.event("discarded_stale_inference", sequence=self.sequence)
                    self.save()
                    return self.state
                warnings = list(dict.fromkeys(view.get("warnings", []) + report.get("warnings", [])))
                if not report.get("exact_training_observation", False):
                    warnings.append("冻结模型的实机迁移建议；仍有未恢复的输入特征，不代表原生评测表现。")
                self.state.update(status="ready", message="请人工执行；操作后按快捷键更新", advice=advice,
                                  warnings=warnings, coverage=report.get("coverage", {
                                      "known_float_count": report.get("known_float_count"),
                                      "float_count": report.get("float_count")}))
                self.advice_key = key
                self.set_pending(view, binding, proposed_policy)
                self.cache[key] = {"advice": advice, "warnings": warnings,
                                   "coverage": self.state["coverage"]}
                self.cached_runtimes[key] = proposed_policy
                # Cache only this session's bounded recent public snapshots.
                if len(self.cache) > 512:
                    old_key = next(iter(self.cache))
                    del self.cache[old_key]
                    del self.cached_runtimes[old_key]
                atomic(folder / "ADVICE.json", advice)
        except Exception as error:
            self.advice_key = None
            self.state.update(status="blocked", advice=None,
                              message=f"暂不建议：{type(error).__name__}: {error}")
            atomic(folder / "FAILED.json", {"type": type(error).__name__, "error": str(error)})
        self.event("refresh", sequence=self.sequence, status=self.state["status"],
                   source=self.source.info(), advice=self.state.get("advice"),
                   message=self.state["message"])
        self.save()
        return self.state

    def set_pending(self, view, binding, policy):
        replay = self.source.replay
        context = {"round": view.get("round"), "epoch": getattr(replay, "epoch", None)}
        if binding.get("kind") != "choose":
            context["options_offset"] = (view.get("last_options") or {}).get("offset")
        self.pending = {"generation": self.source.info().get("generation", 0),
                        "row": len(replay.rows) if replay else 0,
                        "binding": binding, "sequence": self.sequence, "policy": policy,
                        "context": context}

    def describe(self, binding, action):
        if binding.get("text"):
            return {"text": binding["text"], "details": binding.get("selection_caveat") or binding.get("details", ""),
                    "action": action, "binding": binding}
        entity = binding.get("entity") or binding.get("source") or {}
        if not isinstance(entity, dict):
            entity = {}
        name = entity.get("name") or entity.get("card_id") or entity.get("card") or "该对象"
        verbs = {"buy": "购买", "sell": "出售", "play": "打出", "magnetize": "磁力合体",
                 "move": "移动", "activate": "发动", "power": "使用英雄技能",
                 "hero_power": "使用英雄技能", "choose": "选择", "choice": "选择"}
        simple = {"refresh": "刷新酒馆", "upgrade": "升级酒馆", "freeze": "切换酒馆冻结",
                  "end": "结束经营，等待战斗", "dark_gift": "打开黑暗之赐"}
        kind = binding.get("kind", "")
        text = simple.get(kind, verbs.get(kind, kind) + "「" + name + "」")
        if kind in ("move", "play", "magnetize") and "position" in binding:
            text += f"，放到第 {binding['position'] + 1} 位"
        target = binding.get("target")
        if isinstance(target, dict):
            text += " → 「" + (target.get("name") or target.get("card_id") or "目标") + "」"
        if binding.get("targets") and not target:
            raise ValueError("目标尚未解析；不展示不完整动作")
        if binding.get("suboptions") and not binding.get("suboption"):
            raise ValueError("抉择尚未解析；不展示不完整动作")
        return {"text": text, "details": "这是模型建议；未向游戏发送任何输入。",
                "action": action, "binding": binding}

    def navigate(self, delta):
        if self.mode != "replay":
            raise ValueError("Live mode cannot navigate historical frames")
        if delta < 0:
            self.actual_seen.clear()
            self.pending = None
        if self.source.goto(self.source.index + delta):
            # A replay revisit must not inherit a future guard state.
            if self.policy is not None and hasattr(self.policy, "reset"):
                self.policy.reset()
            self.policy_context = None
            self.advice_key = None
            return self.refresh()
        return self.state

    def note(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("标注需为 1–4000 字的文字")
        self.event("human_note", text=text.strip(), source=self.source.info(),
                   advice=self.state.get("advice"))
        return {"ok": True, "message": "标注已保存"}

    def close(self):
        atomic(self.session / "RESULT.json", {"status": "stopped", "started_at": self.started,
            "ended_at": now(), "refreshes": self.sequence, "source": self.source.info(),
            "training": False, "game_inputs_sent": 0, "last_status": self.state["status"]})
        if hasattr(self.source, "close"):
            self.source.close()
