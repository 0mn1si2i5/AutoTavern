//! Observation v2 = the v1 observation (`obs.rs`, unchanged, first) followed by extension
//! blocks that full-content play needs:
//!
//! - which cards are attached to each minion (Magnetic minions, Dark Gifts, granted
//!   effects): the most frequent distinct attached cards with their counts, for the own
//!   board, hand, Tavern, pending options and every opponent's last seen warband;
//! - per-minion counters (script counter, Dark Gift state, temporary buffs, Activate uses,
//!   Blood Gems, gained Deathrattles);
//! - the player's "this game" statistics that scale cards (Volumizer, Beetle, Lobster,
//!   Elemental, spell and Battlecry counts...), Dark Gift and trinket timing, hero power
//!   state;
//! - ongoing player effects, per-card counters and levels, scheduled effects;
//! - opponents' trinkets and hero powers (assumed public, like their heroes).
//!
//! Values are raw (not scaled): the model compresses them. Block names and sizes are
//! exported (`FLOAT_BLOCKS`, `CARD_BLOCKS`) so that Python slices by name.

use crate::actions::{N_BOARD, N_CHOICE, N_HAND, N_SHOP};
use crate::cards::CardId;
use crate::engine::Game;
use crate::obs::{N_OPP, N_QUESTS, OBS_CARDS, OBS_FLOATS, Observation, observe_view};
use crate::state::{Item, Minion, Pending, Player};
use crate::types::{Kw, PlayerId, Races};

/// Distinct attached cards reported per own minion / per item / per opponent minion.
pub const K_ATT_BOARD: usize = 6;
pub const K_ATT_ITEM: usize = 2;
pub const K_ATT_OPP: usize = 3;
/// Extra floats per own minion.
pub const X_MINION: usize = 9;
pub const N_STATS: usize = 40;
pub const POWER_F: usize = 7;
pub const G2_F: usize = 20 + N_STATS + 2 * POWER_F;
pub const N_EFFECTS: usize = 8;
pub const N_COUNTERS: usize = 6;
pub const N_LEVELS: usize = 4;
pub const N_SCHED: usize = 4;
pub const K_OPP_TRINKETS: usize = 4;
/// Summary of an opponent (float block `opp_sum`, one row per opponent): what a compact
/// model reads instead of the opponent's minions.
/// - [0..11] its main minion type (one-hot; none without a typed minion) and [11] how many
///   minions of it, / 7: the leaderboard's figure, of the warband of the last completed
///   round, publicly visible for every opponent;
/// - what the player saw in its last combat against the opponent (zero before the first):
///   [12] seen, [13] turns since / 10, [14..25] minions per type / 7, and the strength of
///   that warband: [25] minions / 7, [26] total Attack / 10, [27] total Health / 10,
///   [28] Divine Shields / 7, [29] Venomous or Poisonous / 7, [30] Reborn / 7;
/// - its last combat, on the leaderboard for everybody: [31] won, [32] lost, [33] damage
///   dealt (> 0) or taken (< 0) / 10, [34] combats won (> 0) or lost (< 0) in a row / 5.
pub const OPP_SUM_F: usize = 35;
/// Where the columns of `opp_sum` about strength / recent results start (a model picks
/// what it reads): types first, then strength, then results.
pub const OPP_SUM_TYPES: usize = 25;
pub const OPP_SUM_RESULTS: usize = 31;
/// The player's own last combat (float block `last_combat`): damage dealt (> 0) or taken
/// (< 0) / 10, combats won (> 0) or lost (< 0) in a row / 5.
pub const LAST_COMBAT_F: usize = 2;
/// Reserved positioning metadata; zero in the public player observation.
pub const PHASE_F: usize = 2;
/// Reserved policy conditioning; zero means unrestricted normal play.
pub const COND_F: usize = 32;
/// Reserved goal vocabulary slots; zero in normal play.
pub const GOAL_K: usize = 8;
/// Reserved item flags; zero in normal play.
pub const ITEM_SLOTS: usize = N_BOARD + N_HAND + N_SHOP + N_CHOICE;
pub const ITEM_KEY_F: usize = 2 * ITEM_SLOTS;
/// Index of the first item slot (the board) in the card array (`obs::OBS_CARDS` order).
pub const ITEM_CARDS_AT: usize =
    1 + crate::actions::N_POWERS + 2 + 3 * N_QUESTS + crate::obs::N_SECRETS;

pub const FLOAT_BLOCKS: [(&str, usize); 17] = [
    ("g2", G2_F),
    ("board_x", N_BOARD * X_MINION),
    ("board_att_n", N_BOARD * K_ATT_BOARD),
    ("hand_att_n", N_HAND * K_ATT_ITEM),
    ("shop_att_n", N_SHOP * K_ATT_ITEM),
    ("choice_att_n", N_CHOICE * K_ATT_ITEM),
    ("oppb_att_n", N_OPP * N_BOARD * K_ATT_OPP),
    ("effects_n", N_EFFECTS),
    ("counters_n", N_COUNTERS),
    ("levels_n", N_LEVELS),
    ("sched_f", N_SCHED * 3),
    ("quest_data", N_QUESTS),
    ("last_combat", LAST_COMBAT_F),
    ("opp_sum", N_OPP * OPP_SUM_F),
    ("phase", PHASE_F),
    ("item_key", ITEM_KEY_F),
    ("cond", COND_F),
];

pub const CARD_BLOCKS: [(&str, usize); 13] = [
    ("board_att", N_BOARD * K_ATT_BOARD),
    ("hand_att", N_HAND * K_ATT_ITEM),
    ("shop_att", N_SHOP * K_ATT_ITEM),
    ("choice_att", N_CHOICE * K_ATT_ITEM),
    ("oppb_att", N_OPP * N_BOARD * K_ATT_OPP),
    ("opp_trinkets", N_OPP * K_OPP_TRINKETS),
    ("opp_power", N_OPP),
    ("effects", N_EFFECTS),
    ("counters", N_COUNTERS),
    ("levels", N_LEVELS),
    ("sched", N_SCHED),
    ("last_spell", 1),
    ("goal_cards", GOAL_K),
];

const fn total(blocks: &[(&str, usize)]) -> usize {
    let mut s = 0;
    let mut i = 0;
    while i < blocks.len() {
        s += blocks[i].1;
        i += 1;
    }
    s
}

pub const OBS2_FLOATS: usize = OBS_FLOATS + total(&FLOAT_BLOCKS);
pub const OBS2_CARDS: usize = OBS_CARDS + total(&CARD_BLOCKS);

fn card_index(c: CardId) -> i32 {
    c.idx() as i32 + 1
}

/// Values which refer to allocated entities are represented by local slot + 1.
/// Other powers keep their visible numerical counters / player-seat choices.
pub fn visible_power_values(pl: &Player, h: &crate::state::HeroPowerState) -> Vec<i32> {
    let name = crate::cards::db().get(h.card).id.as_str();
    h.values
        .iter()
        .map(|v| match name {
            "TB_BaconShop_HP_101" => pl
                .shop
                .iter()
                .position(|s| s.sid as i32 == *v)
                .map_or(0, |i| i as i32 + 1),
            "TB_BaconShop_HP_042" => pl
                .board
                .iter()
                .position(|m| m.eid as i32 == *v)
                .map_or(0, |i| i as i32 + 1),
            _ => *v,
        })
        .collect()
}

/// The `k` most frequent distinct attached cards of a minion with their counts.
fn top_attached(m: Option<&Minion>, k: usize) -> Vec<(CardId, u32)> {
    let mut out: Vec<(CardId, u32)> = Vec::new();
    if let Some(m) = m {
        for a in &m.attached {
            match out.iter_mut().find(|x| x.0 == *a) {
                Some(x) => x.1 += 1,
                None => out.push((*a, 1)),
            }
        }
        out.sort_by_key(|a| std::cmp::Reverse(a.1));
        out.truncate(k);
    }
    out
}

fn push_att(f: &mut Vec<f32>, c: &mut Vec<i32>, m: Option<&Minion>, k: usize) {
    let top = top_attached(m, k);
    for i in 0..k {
        match top.get(i) {
            Some((card, n)) => {
                c.push(card_index(*card));
                f.push(*n as f32);
            }
            None => {
                c.push(0);
                f.push(0.0);
            }
        }
    }
}

/// Minions of each type on a warband (a minion of all types counts for each).
fn type_counts(board: &[Minion]) -> [u8; Races::COUNT] {
    let mut n = [0u8; Races::COUNT];
    for m in board {
        for (i, x) in n.iter_mut().enumerate() {
            *x += (m.races.0 >> i & 1) as u8;
        }
    }
    n
}

/// One row of the float block `opp_sum` (`OPP_SUM_F`): opponent `o` as `p` knows it.
fn push_opp_summary(f: &mut Vec<f32>, g: &Game, p: PlayerId, o: PlayerId, all: bool) {
    if g.phase == crate::engine::Phase::HeroSelect && !all {
        f.extend([0.0; OPP_SUM_F]);
        return;
    }
    // The leaderboard: main type and its count, of the warband of the last round.
    let public = type_counts(&g.players[o].seen_board);
    let top = (0..Races::COUNT).max_by_key(|i| (public[*i], Races::COUNT - i));
    let main = top.filter(|i| public[*i] > 0);
    for i in 0..Races::COUNT {
        f.push((main == Some(i)) as u8 as f32);
    }
    f.push(main.map_or(0, |i| public[i]) as f32 / 7.0);
    // The player's own last combat against it.
    let (board, turn) = g.board_shown(p, o, all);
    let seen = turn > 0 || !board.is_empty();
    f.push(seen as u8 as f32);
    f.push(if seen {
        g.turn.saturating_sub(turn) as f32 / 10.0
    } else {
        0.0
    });
    for n in type_counts(board) {
        f.push(n as f32 / 7.0);
    }
    let count = |k: Kw| board.iter().filter(|m| m.kw.has(k)).count();
    let venom = board
        .iter()
        .filter(|m| m.kw.has(Kw::VENOMOUS) || m.kw.has(Kw::POISONOUS))
        .count();
    f.extend([
        board.len() as f32 / 7.0,
        board.iter().map(|m| i64::from(m.atk)).sum::<i64>() as f32 / 10.0,
        board.iter().map(|m| i64::from(m.hp)).sum::<i64>() as f32 / 10.0,
        count(Kw::DIVINE_SHIELD) as f32 / 7.0,
        venom as f32 / 7.0,
        count(Kw::REBORN) as f32 / 7.0,
    ]);
    // Its last combat (the leaderboard's recent results).
    let st = &g.players[o].stats;
    f.extend([
        st.won_last as u8 as f32,
        st.lost_last as u8 as f32,
        st.last_damage as f32 / 10.0,
        st.streak as f32 / 5.0,
    ]);
}

fn item_minion(it: Option<&Item>) -> Option<&Minion> {
    match it {
        Some(Item::Minion(m)) => Some(m),
        _ => None,
    }
}

/// The minion behind pending option `i` (Discover / Timewarp items, target minions).
fn option_minion(pl: &Player, i: usize) -> Option<&Minion> {
    match pl.pending.last()? {
        Pending::Discover { options, .. } => item_minion(options.get(i)),
        Pending::Timewarp { options, .. } => item_minion(options.get(i).map(|o| &o.0)),
        Pending::Target { options, .. } => {
            let e = *options.get(i)?;
            pl.board
                .iter()
                .find(|m| m.eid == e)
                .or_else(|| item_minion(pl.shop.iter().find(|s| s.sid == e).map(|s| &s.item)))
                .or_else(|| item_minion(pl.hand.iter().find(|h| h.hid == e).map(|h| &h.item)))
        }
        _ => None,
    }
}

fn push_pairs(
    f: &mut Vec<f32>,
    c: &mut Vec<i32>,
    xs: impl Iterator<Item = (CardId, i32)>,
    n: usize,
) {
    let mut k = 0;
    for (card, v) in xs.take(n) {
        c.push(card_index(card));
        f.push(v as f32);
        k += 1;
    }
    for _ in k..n {
        c.push(0);
        f.push(0.0);
    }
}

pub fn observe2(g: &Game, p: PlayerId) -> Observation {
    observe2_view(g, p, g.players[p].sees_all_boards)
}

/// Observation v2 with the opponents' warbands in a given view (`Game::board_shown`).
pub fn observe2_view(g: &Game, p: PlayerId, all: bool) -> Observation {
    let mut o = observe_view(g, p, all);
    o.floats.reserve(OBS2_FLOATS - OBS_FLOATS);
    o.cards.reserve(OBS2_CARDS - OBS_CARDS);
    let pl = &g.players[p];
    let cfg = &g.cfg;
    let b = |x: bool| x as u8 as f32;

    // ---- g2: floats only
    {
        let f = &mut o.floats;
        let start = f.len();
        f.push(pl.actions_this_turn as f32);
        f.push(pl.stats.gold_spent_turn as f32);
        // Dark Gift button.
        f.push(cfg.dark_gift_uses.saturating_sub(pl.dark_gift_uses) as f32);
        f.push(cfg.dark_gift_cost as f32);
        f.push(b(
            g.turn >= cfg.dark_gift_unlock_turn && cfg.dark_gift_uses > 0
        ));
        f.push(b(pl.dark_gift_used_turn));
        let tiers = cfg.dark_gift_tiers(g.turn);
        for t in 1..=6u8 {
            f.push(b(cfg.dark_gift_uses > 0 && tiers.contains(&t)));
        }
        // Trinket offers: turns until the Lesser / Greater offer (0 = passed or off).
        let until = |turn: u32| {
            if turn >= g.turn && turn - g.turn <= 8 {
                (turn - g.turn + 1) as f32
            } else {
                0.0
            }
        };
        let greater = if pl.greater_turn > 0 {
            pl.greater_turn
        } else {
            cfg.greater_trinket_turn
        };
        f.push(until(cfg.lesser_trinket_turn));
        f.push(if pl.greater_offered {
            0.0
        } else {
            until(greater)
        });
        f.push(b(pl.greater_offered));
        f.push(b(g.can_upgrade(p)));
        f.push(b(pl.tier7));
        f.push(b(pl.skipping));
        f.push(b(pl.deity.as_ref().is_some_and(|d| d.golden)));
        f.push(b(pl.deity.as_ref().is_some_and(|d| d.reborn)));
        let s = &pl.stats;
        f.extend(
            [
                s.ballers_sold,
                s.spells_cast,
                s.tavern_spells_cast,
                s.battlecries_triggered,
                s.deathrattles_triggered,
                s.elementals_played,
                s.cards_played,
                s.cards_discarded,
                s.golden_played,
                s.minions_bought,
                s.cards_bought,
                s.minions_sold,
                s.gold_spent_total,
                s.refreshes,
                s.undead_atk,
                s.beetle_atk,
                s.beetle_hp,
                s.eternal_knight_deaths,
                s.knight_bonus,
                s.elementals_played_turn,
                s.battlecries_turn,
                s.spellcraft_cast_turn,
                s.spellcraft_cast,
                s.volumizer_atk,
                s.volumizer_hp,
                s.elem_give_atk,
                s.elem_give_hp,
                s.tavern_elemental_atk,
                s.tavern_elemental_hp,
                s.baller_bonus,
                s.lobster_bonus,
                s.combats_won,
                s.combats_lost,
                s.tied_last as i32,
                s.damage_taken_outside_combat,
                s.spell_discount_next,
                s.spells_cast_turn,
                s.tavern_spells_cast_turn,
                s.cards_played_turn,
                s.minions_played_turn,
            ]
            .map(|x| x as f32),
        );
        for slot in 0..2 {
            match pl.powers.get(slot) {
                Some(h) => f.extend([
                    h.uses_turn as f32,
                    h.counter as f32,
                    h.cost_mod as f32,
                    h.cooldown as f32,
                    h.level as f32,
                    visible_power_values(pl, h).first().copied().unwrap_or(0) as f32,
                    visible_power_values(pl, h).get(1).copied().unwrap_or(0) as f32,
                ]),
                None => f.extend([0.0; POWER_F]),
            }
        }
        debug_assert_eq!(f.len() - start, G2_F);
    }

    let (f, c) = (&mut o.floats, &mut o.cards);
    // ---- own board: extra floats, then attached cards (floats: counts)
    for i in 0..N_BOARD {
        match pl.board.get(i) {
            Some(m) => f.extend([
                m.attached.len() as f32,
                m.counter as f32,
                m.gift_data as f32,
                m.temp_atk as f32,
                m.temp_hp as f32,
                m.activations as f32,
                m.gems as f32,
                m.gained_dr.len() as f32,
                b(m.triple_reward),
            ]),
            None => f.extend([0.0; X_MINION]),
        }
    }
    for i in 0..N_BOARD {
        push_att(f, c, pl.board.get(i), K_ATT_BOARD);
    }
    for i in 0..N_HAND {
        push_att(
            f,
            c,
            item_minion(pl.hand.get(i).map(|h| &h.item)),
            K_ATT_ITEM,
        );
    }
    for i in 0..N_SHOP {
        push_att(
            f,
            c,
            item_minion(pl.shop.get(i).map(|s| &s.item)),
            K_ATT_ITEM,
        );
    }
    for i in 0..N_CHOICE {
        push_att(f, c, option_minion(pl, i), K_ATT_ITEM);
    }
    // ---- opponents (same order as v1): attached cards of their last seen warbands
    let n = g.players.len();
    let others: Vec<PlayerId> = (1..n).map(|k| (p + k) % n).take(N_OPP).collect();
    for i in 0..N_OPP {
        let board = others.get(i).map(|o| g.board_shown(p, *o, all).0);
        for j in 0..N_BOARD {
            push_att(f, c, board.and_then(|b| b.get(j)), K_ATT_OPP);
        }
    }
    for i in 0..N_OPP {
        // Trinkets are on the leaderboard.
        let known = others
            .get(i)
            .filter(|_| g.phase != crate::engine::Phase::HeroSelect)
            .map(|o| &g.players[*o].trinkets);
        for k in 0..K_OPP_TRINKETS {
            c.push(known.and_then(|t| t.get(k)).map_or(0, |t| card_index(*t)));
        }
    }
    for i in 0..N_OPP {
        let op = others
            .get(i)
            .filter(|_| all || g.phase != crate::engine::Phase::HeroSelect)
            .map(|o| &g.players[*o]);
        c.push(
            op.and_then(|o| o.powers.first())
                .map_or(0, |h| card_index(h.card)),
        );
    }
    // ---- ongoing effects, counters, levels, scheduled effects
    push_pairs(
        f,
        c,
        pl.effects.iter().map(|e| (e.card, e.stacks)),
        N_EFFECTS,
    );
    // The per-card counters of what the player holds (trinkets, hero powers, quests). The
    // engine also keeps the turn every minion card was last played in there, under the
    // card's id with the copy mark (`recruit::remember_play`): a record of the rules, not a
    // counter a player sees. It is left out - it would fill the block from the first minion
    // played and push the real counters (a trinket's count-down, its state) out of it.
    let shown = pl
        .counters
        .iter()
        .copied()
        .filter(|x| x.0.0 & CardId::COPY == 0 || !crate::cards::db().get(x.0).is_minion());
    push_pairs(f, c, shown, N_COUNTERS);
    push_pairs(f, c, pl.levels.iter().copied(), N_LEVELS);
    for i in 0..N_SCHED {
        match pl.scheduled.get(i) {
            Some(s) => {
                c.push(card_index(s.script));
                f.extend([s.turns_left as f32, s.a as f32, s.b as f32]);
            }
            None => {
                c.push(0);
                f.extend([0.0; 3]);
            }
        }
    }
    for i in 0..N_QUESTS {
        f.push(pl.quests.get(i).map_or(0, |q| q.data) as f32);
    }
    f.extend([
        pl.stats.last_damage as f32 / 10.0,
        pl.stats.streak as f32 / 5.0,
    ]);
    for i in 0..N_OPP {
        match others.get(i) {
            Some(o) => push_opp_summary(f, g, p, *o, all),
            None => f.extend([0.0; OPP_SUM_F]),
        }
    }
    c.push(pl.stats.last_spell_cast.map_or(0, card_index));
    c.extend([0; GOAL_K]);
    f.extend([0.0; PHASE_F]);
    f.extend([0.0; ITEM_KEY_F]);
    f.extend([0.0; COND_F]);
    debug_assert_eq!(o.floats.len(), OBS2_FLOATS);
    debug_assert_eq!(o.cards.len(), OBS2_CARDS);
    o
}
