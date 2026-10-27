//! Complete player-visible JSON, independent of any model or training framework.
//! Hidden opponent zones, RNG, seeds, event traces and replay frames are never exported.
use crate::debug::{card_json, minion_json};
use crate::{Game, cards::CardId, engine::Phase};
use serde_json::{Value, json};

pub fn player_view(g: &Game, p: usize) -> Value {
    let source = crate::debug::public_view_without_combats(g, p);
    let mut view = serde_json::Map::new();
    for key in [
        "seat",
        "turn",
        "phase",
        "version",
        "lobby_races",
        "has_aberrations",
        "deity",
        "paradox",
        "alive",
        "me",
        "mask",
        "actions",
    ] {
        if let Some(value) = source.get(key) {
            view.insert(key.into(), value.clone());
        }
    }
    let pl = &g.players[p];
    if let Some(me) = view.get_mut("me").and_then(Value::as_object_mut) {
        // Keep an explicit owner-field allowlist as the development view evolves.
        me.retain(|key, _| {
            [
                "hero",
                "hero_power",
                "hero_powers",
                "health",
                "armor",
                "alive",
                "placement",
                "win_streak",
                "last_damage",
                "last_warband",
                "tier",
                "gold",
                "max_gold",
                "max_gold_bonus",
                "gold_next_turn",
                "upgrade_price",
                "can_upgrade_tier",
                "refresh_price",
                "minion_price",
                "free_refreshes",
                "frozen",
                "ended_turn",
                "board",
                "hand",
                "shop",
                "trinkets",
                "quests",
                "rewards",
                "secrets",
                "hand_queue",
                "recent_opponents",
                "buddy_button",
                "chronum_balance",
                "hand_limit",
                "buddy",
                "casting",
                "cancellable",
                "dark_gift",
                "deity",
                "effects",
                "scheduled",
                "stats",
                "pending",
                "last_opponent",
                "next_opponent",
                "next_is_ghost",
            ]
            .contains(&key.as_str())
        });
        if let Some(crate::state::Pending::Target { options, .. }) = pl.pending.last() {
            me["pending"]["targets"] = json!(
                options
                    .iter()
                    .enumerate()
                    .map(|(i, id)| {
                        let location = pl
                            .board
                            .iter()
                            .position(|m| m.eid == *id)
                            .map(|slot| json!({"zone":"board","slot":slot}))
                            .or_else(|| {
                                pl.shop
                                    .iter()
                                    .position(|s| s.sid == *id)
                                    .map(|slot| json!({"zone":"shop","slot":slot}))
                            })
                            .or_else(|| {
                                pl.hand
                                    .iter()
                                    .position(|h| h.hid == *id)
                                    .map(|slot| json!({"zone":"hand","slot":slot}))
                            })
                            .unwrap_or(json!({"zone":"hero"}));
                        json!({"choice":i,"action":crate::actions::A_CHOOSE+i,"location":location})
                    })
                    .collect::<Vec<_>>()
            );
        }
        for (key, minions) in [
            ("board", pl.board.iter().map(Some).collect::<Vec<_>>()),
            ("hand", pl.hand.iter().map(|h| h.item.as_minion()).collect()),
            ("shop", pl.shop.iter().map(|s| s.item.as_minion()).collect()),
            (
                "hand_queue",
                pl.hand_queue.iter().map(|i| i.as_minion()).collect(),
            ),
        ] {
            if let Some(values) = me.get_mut(key).and_then(Value::as_array_mut) {
                for (value, minion) in values.iter_mut().zip(minions) {
                    if let Some(m) = minion {
                        value["gift_data"] = json!(m.gift_data);
                        value["gems"] = json!(m.gems);
                        value["gained_deathrattles"] = json!(
                            m.gained_dr
                                .iter()
                                .map(|c| card_json(*c))
                                .collect::<Vec<_>>()
                        );
                        value["doomed_turn"] = json!(m.doomed_turn);
                    }
                }
            }
        }
        me.insert("trinket_timing".into(),json!({"lesser_turn":g.cfg.lesser_trinket_turn,
            "greater_turn":if pl.greater_turn>0 {pl.greater_turn}else{g.cfg.greater_trinket_turn},"greater_offered":pl.greater_offered}));
        me.insert(
            "counters".into(),
            json!(
                pl.counters
                    .iter()
                    .filter(
                        |(c, _)| c.0 & CardId::COPY == 0 || !crate::cards::db().get(*c).is_minion()
                    )
                    .map(|(c, n)| json!({"card":card_json(*c),"value":n}))
                    .collect::<Vec<_>>()
            ),
        );
        me.insert(
            "levels".into(),
            json!(
                pl.levels
                    .iter()
                    .map(|(c, n)| json!({"card":card_json(*c),"level":n}))
                    .collect::<Vec<_>>()
            ),
        );
        me.insert("hero_power_state".into(), json!(pl.powers.iter().map(|h| json!({"card":card_json(h.card),"uses_turn":h.uses_turn,"uses_game":h.uses_game,"counter":h.counter,"cost_mod":h.cost_mod,"cooldown":h.cooldown,"level":h.level,"values":crate::obs2::visible_power_values(pl,h)})).collect::<Vec<_>>()));
    }
    let selecting = g.phase == Phase::HeroSelect;
    let opponents = g.players.iter().filter(|o| o.id != p).map(|o| {
        let (board,turn) = g.board_shown(p,o.id,pl.sees_all_boards);
        let original = source["opponents"].as_array().unwrap().iter().find(|x| x["id"] == json!(o.id)).unwrap();
        if selecting {
            return json!({"id":o.id,"hero":null,"health":40,"armor":0,"tier":1,"alive":true,"placement":0,
                "trinkets":[],"hero_powers":[],"quests":[],"rewards":[],"seen_board":[],"seen_turn":0,
                "last_combat":null,"last_warband":null});
        }
        json!({"id":o.id,"hero":o.hero.map(card_json),"health":o.health,"armor":o.armor,"tier":o.tier,"alive":o.alive,"placement":o.placement,
            "trinkets":o.trinkets.iter().map(|c|card_json(*c)).collect::<Vec<_>>(),
            "hero_powers":o.powers.iter().map(|h|card_json(h.card)).collect::<Vec<_>>(),
            "quests":original["quests"],"rewards":original["rewards"],"last_warband":original["last_warband"],
            "seen_board":board.iter().map(minion_json).collect::<Vec<_>>(),"seen_turn":turn,
            "was_last_opponent":pl.last_opponent==Some(o.id),"is_next_opponent":pl.next_opponent==Some(o.id),
            "last_combat":{"won":o.stats.won_last,"lost":o.stats.lost_last,"tied":o.stats.tied_last,"damage":o.stats.last_damage,"streak":o.stats.streak}})
    }).collect::<Vec<_>>();
    view.insert("opponents".into(), json!(opponents));
    // Stable slot indices carry actions; internal entity allocation must not reveal
    // how much work another seat has done during the asynchronous recruit phase.
    if let Some(me) = view.get_mut("me").and_then(Value::as_object_mut) {
        let casting = me.get("casting").and_then(Value::as_u64);
        let hand_index = casting.and_then(|id| pl.hand.iter().position(|h| u64::from(h.hid) == id));
        me.remove("casting");
        me.insert("casting_hand_index".into(), json!(hand_index));
        me.insert(
            "scheduled".into(),
            json!(
                pl.scheduled
                    .iter()
                    .map(|s| json!({
                        "card":card_json(s.script),"turns_left":s.turns_left,"if_won":s.if_won,
                        "a":s.a,"b":s.b,
                        "target_board_index":pl.board.iter().position(|m|m.eid==s.target)
                    }))
                    .collect::<Vec<_>>()
            ),
        );
    }
    let mut result = Value::Object(view);
    strip_internal(&mut result);
    result
}

fn strip_internal(value: &mut Value) {
    match value {
        Value::Object(fields) => {
            for key in [
                "eid",
                "hid",
                "sid",
                "link",
                "group",
                "pool_copies",
                "pool_parts",
                "implemented",
            ] {
                fields.remove(key);
            }
            for child in fields.values_mut() {
                strip_internal(child);
            }
        }
        Value::Array(values) => {
            for child in values {
                strip_internal(child);
            }
        }
        _ => (),
    }
}
