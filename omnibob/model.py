"""Small observe2 entity Transformer with structured 245-action pointer heads.

Independent implementation, not TavernAgent. No rule-text embeddings or calibrated
opponent library. Opponent tokens use only the API's normally visible summaries.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import taverngym as tg

from .observation_v2 import extension_slices


@dataclass
class ModelConfig:
    width: int = 64
    layers: int = 2
    heads: int = 4
    ff_multiplier: int = 2


def symlog(value):
    return value.sign() * torch.log1p(value.abs())


class EntityActorCritic(nn.Module):
    observation_version = 2

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        d = self.config.width
        if d % self.config.heads:
            raise ValueError("width must be divisible by attention heads")
        self.fs, self.cs = extension_slices()
        self.embedding = nn.Embedding(tg.N_CARDS, d, padding_idx=0)
        width, raw = tg.card_static()
        static = np.frombuffer(raw, dtype="<f4").reshape(tg.N_CARDS, width).copy()
        self.register_buffer("card_static", torch.tensor(static))
        self.static_projection = nn.Linear(width, d, bias=False)
        self.board = nn.Linear(40, d)
        self.item = nn.Linear(40, d)
        self.opponent = nn.Linear(44, d)
        self.opp_minion = nn.Linear(31, d)
        self.global_blocks = ("g2", "effects_n", "counters_n", "levels_n", "sched_f", "quest_data", "last_combat")
        global_size = 68 + sum(self.fs[n].stop-self.fs[n].start for n in self.global_blocks)
        self.global_projection = nn.Linear(global_size, d)
        # Distinct slot embeddings preserve board/hand/shop/choice position and zone identity.
        self.positions = nn.Parameter(torch.randn(1, 49, d) * .02)
        layer = nn.TransformerEncoderLayer(d, self.config.heads, d*self.config.ff_multiplier,
                                          dropout=0, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, self.config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.general = nn.Linear(d, 8)
        self.buy = nn.Linear(d*2, 1)
        self.sell = nn.Linear(d*2, 1)
        self.activate = nn.Linear(d*2, 1)
        self.choose = nn.Linear(d*2, 1)
        self.play = nn.Sequential(nn.Linear(d*3, d), nn.Tanh(), nn.Linear(d, 1))
        self.magnetize = nn.Sequential(nn.Linear(d*3, d), nn.Tanh(), nn.Linear(d, 1))
        self.move = nn.Sequential(nn.Linear(d*3, d), nn.Tanh(), nn.Linear(d, 1))
        self.empty_position = nn.Parameter(torch.zeros(1, 1, d))
        self.value = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        # Small initial actor weights avoid arbitrary confidence in one action family.
        for module in (self.general, self.buy, self.sell, self.activate, self.choose,
                       self.play[-1], self.magnetize[-1], self.move[-1]):
            nn.init.orthogonal_(module.weight, gain=.01)
            nn.init.zeros_(module.bias)

    def cards(self, indices):
        # Padding stays zero even though linear projections of a real card can be nonzero.
        return (self.embedding(indices) + self.static_projection(symlog(self.card_static[indices]))) * (indices != 0).unsqueeze(-1)

    def pooled_cards(self, indices):
        values = self.cards(indices)
        return values.sum(-2) / (indices != 0).sum(-1, keepdim=True).clamp_min(1)

    def attached(self, f, c, name, entities, count):
        ids = c[:, self.cs[name]].reshape(-1, entities, count)
        counts = f[:, self.fs[name+"_n"]].reshape(-1, entities, count)
        values = self.cards(ids) * torch.log1p(counts.clamp_min(0)).unsqueeze(-1)
        return values.sum(-2) / (ids != 0).sum(-1, keepdim=True).clamp_min(1)

    def forward(self, f, c, mask):
        batch = f.shape[0]
        board_f = f[:, 68:285].reshape(-1, 7, 31)
        hand_f = f[:, 285:685].reshape(-1, 10, 40)
        shop_f = f[:, 685:1005].reshape(-1, 8, 40)
        choice_f = f[:, 1005:1645].reshape(-1, 16, 40)
        opp_f = f[:, 1645:1708].reshape(-1, 7, 9)
        board = self.board(symlog(torch.cat((board_f, f[:, self.fs["board_x"]].reshape(-1, 7, 9)), -1)))
        board = board + self.cards(c[:, 19:26]) + self.attached(f, c, "board_att", 7, 6)
        items = []
        for values, ids, name, count in ((hand_f, c[:, 26:36], "hand_att", 10),
                                         (shop_f, c[:, 36:44], "shop_att", 8),
                                         (choice_f, c[:, 44:60], "choice_att", 16)):
            items.append(self.item(symlog(values)) + self.cards(ids) + self.attached(f, c, name, count, 2))
        opponent = self.opponent(symlog(torch.cat((opp_f, f[:, self.fs["opp_sum"]].reshape(-1, 7, 35)), -1)))
        seen = f[:, 1708:3227].reshape(-1, 7, 7, 31)
        seen_tokens = self.opp_minion(symlog(seen)) + self.cards(c[:, 75:124].reshape(-1, 7, 7))
        seen_tokens += self.attached(f, c, "oppb_att", 49, 3).reshape(batch, 7, 7, -1)
        seen_mask = seen[..., 0:1]
        opponent += (seen_tokens*seen_mask).sum(2) / seen_mask.sum(2).clamp_min(1)
        opponent += self.cards(c[:, 68:75]) + self.cards(c[:, self.cs["opp_power"]])
        opponent += self.pooled_cards(c[:, self.cs["opp_trinkets"]].reshape(batch, 7, 4))
        glob = self.global_projection(symlog(torch.cat([f[:, :68]] + [f[:, self.fs[n]] for n in self.global_blocks], -1)))
        global_ids = torch.cat([c[:, :19], c[:, 60:68]] + [c[:, self.cs[n]] for n in
                               ("effects", "counters", "levels", "sched", "last_spell")], -1)
        glob += self.pooled_cards(global_ids)
        tokens = torch.cat([glob[:, None], board, *items, opponent], dim=1) + self.positions
        present = torch.cat([torch.ones((batch, 1), dtype=torch.bool, device=f.device),
                             board_f[..., 0] > 0, hand_f[..., 0] > 0, shop_f[..., 0] > 0,
                             choice_f[..., 0] > 0, opp_f[..., 0] > 0], dim=1)
        tokens = self.norm(self.transformer(tokens, src_key_padding_mask=~present))
        glob, board, hand, shop, choices = (tokens[:, 0], tokens[:, 1:8], tokens[:, 8:18],
                                          tokens[:, 18:26], tokens[:, 26:42])

        def one(head, entities):
            return head(torch.cat((entities, glob[:, None].expand(-1, entities.shape[1], -1)), -1)).squeeze(-1)

        def pair(head, sources, targets):
            ns, nt = sources.shape[1], targets.shape[1]
            pairs = torch.cat((sources[:, :, None].expand(-1, -1, nt, -1),
                               targets[:, None].expand(-1, ns, -1, -1),
                               glob[:, None, None].expand(-1, ns, nt, -1)), -1)
            return head(pairs).reshape(batch, ns*nt)

        positions = torch.cat((board, self.empty_position.expand(batch, -1, -1)), 1)
        logits = torch.cat((self.general(glob), one(self.buy, shop), one(self.sell, board),
                            one(self.activate, board), one(self.choose, choices), pair(self.play, hand, positions),
                            pair(self.magnetize, hand, board), pair(self.move, board, board)), -1)
        return logits.masked_fill(~mask, -1e9), self.value(glob).squeeze(-1)
