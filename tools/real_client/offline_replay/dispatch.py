"""Join the archived system-input receipts to sends without giving credit for time alone."""

import json
from datetime import datetime


def seconds_local(timestamp):
    return (timestamp + 8 * 3600) % 86400


def join_actions(directory, rows):
    used = set()
    report = []
    for line in (directory / "actions.jsonl").read_text().splitlines():
        attempt = json.loads(line)
        step, label = attempt["step"], attempt["action"]["label"]
        folder = directory / "steps" / f"{step:06d}"
        view = json.loads((folder / "VISIBLE_VIEW.json").read_text())
        choice = label in ("model_choice", "confirm_trinket")
        decision = folder / ("MODEL_DECISION_choice.json" if choice else "MODEL_DECISION.json")
        binding = json.loads(decision.read_text())["binding"]
        entity = binding["entity"]["entity_id"]
        when = datetime.fromisoformat(attempt["utc"]).timestamp()
        request = json.loads((folder / (label + ".json")).read_text())
        # Use the archived request and receipt, not mutable copy-time filesystem mtimes.
        start = request["deadline_unix"] - 5
        if not 0 <= when - start < 5:
            raise ValueError("input outside archived request window")
        candidates = []
        for row in rows:
            delta = (row["seconds"] - seconds_local(when) + 43200) % 86400 - 43200
            if start - when <= delta <= 0.4:
                candidates.append(row)
        item = dict(
            step=step,
            label=label,
            intended_entity=entity,
            prefix=view["source_bytes"],
            intended_position=binding.get("position"),
            choice=view.get("active_choice") and view["active_choice"]["id"],
        )
        if len(candidates) != 1:
            item["input_status"] = "no_send" if not candidates else "ambiguous_sends"
        else:
            row = candidates[0]
            send = row["binding"]
            item.update(
                row=row["row"],
                offset=row["offset"],
                actual=send,
                execution_status=row["execution"]["status"],
            )
            if choice:
                matches = (
                    row["event"] == "SendChoices"
                    and send["choice"] == item["choice"]
                    and send["chosen"] == [entity]
                )
            else:
                kind = binding["kind"]
                expected_position = 0
                if kind == "move" or (
                    kind == "play" and binding["entity"].get("card_type") == "MINION"
                ):
                    expected_position = binding["position"] + 1
                # Pre-action option instance and type from the same archived view.
                valid = view["last_options"]

                def candidate(opt):
                    src = opt.get("entity") or {}
                    if kind == "buy":
                        return "DragBuy" in src.get("card_id", "") and str(
                            src.get("tags", {}).get("2442")
                        ) == str(entity)
                    if kind == "sell":
                        return "DragSell" in src.get("card_id", "")
                    return src.get("entity_id") == entity

                indices = {
                    o["index"] for o in valid["options"] if candidate(o) and o["error"] == "NONE"
                }
                matches = (
                    row["event"] == "SendOption"
                    and valid["id"] == row["option_id"]
                    and send["option"] in indices
                    and send["target"] == (entity if kind in ("buy", "sell") else 0)
                    and send["sub"] == -1
                    and send["position"] == expected_position
                    and row["mapping"]["kind"] == kind
                )
            item["input_status"] = (
                "matched" if matches and row["row"] not in used else "binding_mismatch"
            )
            used.add(row["row"])
        if item["input_status"] == "no_send":
            if choice:
                item["context"] = (
                    "trinket_selection_first_click"
                    if label == "model_choice"
                    else "choice_not_sent"
                )
            elif item["choice"] is not None:
                item["context"] = "choice_already_open"
            else:
                item["context"] = "choose_one_attempt_and_subsequent_no_send_interval"
        report.append(item)
    return report
