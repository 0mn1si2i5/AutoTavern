"""Offline inventory only. No model arrays are constructed and missing data stays unknown."""

import csv

LEGACY_F = [
    ("global", 53),
    ("quests", 6),
    ("pending_kind", 9),
    ("board", 217),
    ("hand", 400),
    ("shop", 320),
    ("choice", 640),
    ("opponents", 63),
    ("opponent_boards", 1519),
]
LEGACY_C = [
    ("hero", 1),
    ("powers", 2),
    ("deity", 1),
    ("choice_source", 1),
    ("quests_rewards", 9),
    ("secrets", 5),
    ("board", 7),
    ("hand", 10),
    ("shop", 8),
    ("choice", 16),
    ("trinkets", 8),
    ("opponent_heroes", 7),
    ("opponent_boards", 49),
]
EXTRA_F = [
    ("g2", 74),
    ("board_x", 63),
    ("board_att_n", 42),
    ("hand_att_n", 20),
    ("shop_att_n", 16),
    ("choice_att_n", 32),
    ("oppb_att_n", 147),
    ("effects_n", 8),
    ("counters_n", 6),
    ("levels_n", 4),
    ("sched_f", 12),
    ("quest_data", 3),
    ("last_combat", 2),
    ("opp_sum", 245),
    ("phase", 2),
    ("item_key", 82),
    ("cond", 32),
]
EXTRA_C = [
    ("board_att", 42),
    ("hand_att", 20),
    ("shop_att", 16),
    ("choice_att", 32),
    ("oppb_att", 147),
    ("opp_trinkets", 28),
    ("opp_power", 7),
    ("effects", 8),
    ("counters", 6),
    ("levels", 4),
    ("sched", 4),
    ("last_spell", 1),
    ("goal_cards", 8),
]


def inventory(path):
    labels = [
        "axis",
        "start",
        "end_exclusive",
        "length",
        "block",
        "definition",
        "access",
        "normalization",
        "available_after",
        "evidence_or_unknown",
        "rules",
    ]
    with path.open("x", newline="") as f:
        out = csv.writer(f)
        out.writerow(labels)
        for axis, base, extra, expected in [
            ("float", LEGACY_F, EXTRA_F, 4017),
            ("card", LEGACY_C, EXTRA_C, 447),
        ]:
            offset = 0
            for ext, blocks in [(False, base), (True, extra)]:
                for block, count in blocks:
                    src = "crates/taverngym/src/" + ("obs2.rs" if ext else "obs.rs")
                    mode = "event_accumulation_unverified"
                    unit = "raw values; see per-feature expression in " + src
                    evidence = "No validated one-to-one tag or update rule for every member"
                    when = "after corresponding visible effect is played"
                    if block in (
                        "board",
                        "hand",
                        "shop",
                        "hero",
                        "powers",
                        "global",
                        "pending_kind",
                        "choice",
                    ):
                        mode = "direct_partial"
                        when = "after PT display or ChoiceCardMgr BEGIN"
                        evidence = "own/tavern entity ID, slot, stats or offer; absent tags and semantic mismatches unknown"
                    if block.startswith("opp"):
                        mode = "encounter_memory_or_unknown"
                        when = "only a previously presented combat or proven visible leaderboard"
                        evidence = "Opponent blocks withheld in v0; no unseen lineup, hidden hand or hover-only fields"
                    if block in (
                        "phase",
                        "item_key",
                        "goal_cards",
                        "cond",
                        "sched",
                        "sched_f",
                        "levels",
                        "levels_n",
                    ):
                        mode = "engine_only_or_unknown"
                        when = "not provided by replay"
                        evidence = "reserved or engine scheduling/levels require explicit contract"
                    if block in ("g2", "board_x", "quests", "quest_data"):
                        mode = "visible_tags_and_accumulation_partial"
                    if axis == "card":
                        unit = "pinned vocabulary index+1; no index emitted here"
                    elif block in ("board", "opponent_boards"):
                        unit = "presence and flags 0/1; atk/hp/max_hp /10; tier /6; attachment/shield counts /3"
                    elif block in ("hand", "shop", "choice"):
                        unit = "item header flags, cost /5, minion encoding or spell data /6"
                    elif block == "global":
                        unit = "mixed; turn /20, alive /8, lengths /7 and /10; observe_view global expressions"
                    elif block == "quests":
                        unit = "presence, progress / max(goal,1)"
                    elif block == "pending_kind":
                        unit = "one-hot, 9 kinds"
                    elif block == "last_combat":
                        unit = "signed damage /10, streak /5"
                    elif block == "opp_sum":
                        unit = "flags, counts /7, turn age /10, atk/hp/damage /10, streak /5"
                    out.writerow(
                        [
                            axis,
                            offset,
                            offset + count,
                            count,
                            block,
                            src,
                            mode,
                            unit,
                            when,
                            evidence,
                            "local observe2 contract; archived clients 253216/255790; cross-build parity unverified",
                        ]
                    )
                    offset += count
            if offset != expected:
                raise ValueError("field inventory mismatch")
