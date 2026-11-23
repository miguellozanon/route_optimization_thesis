"""Attention-based constructive policy for the first-mile CVRP, trained with
REINFORCE and a POMO shared baseline.

References
* Kool, van Hoof & Welling (2019), "Attention, Learn to Solve Routing Problems!", ICLR.
* Kwon et al. (2020), "POMO: Policy Optimization with Multiple Optima for RL", NeurIPS.

Architecture
* Encoder: linear embeddings (depot: (x,y); customer: (x,y,q/Q)) followed by
  L transformer layers (multi-head attention + feed-forward, residuals,
  instance normalisation).
* Decoder: at every step the query is [h_last, remaining load]; one
  multi-head "glimpse" over the node embeddings, then a single-head
  compatibility with tanh clipping (C = 10) gives the logits. Infeasible
  nodes (visited, demand > remaining load, depot twice in a row) are masked.
* POMO: for an instance with n customers, n rollouts start from the n
  different first customers; the baseline of each rollout is the mean return
  of the n rollouts of the same instance (no critic, low variance).
* Inference: greedy decoding of all n starts x 8 symmetric augmentations of
  the unit square; the best tour is returned.

This module needs ``torch`` (optional dependency: ``pip install brlo[neural]``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class InstanceNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(d, affine=True, track_running_stats=False)

    def forward(self, x):            # x: [B, N, d]
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1 = InstanceNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.n2 = InstanceNorm(d)

    def forward(self, h):
        h = self.n1(h + self.mha(h, h, h, need_weights=False)[0])
        return self.n2(h + self.ff(h))


class AttentionPolicy(nn.Module):
    def __init__(self, d: int = 128, heads: int = 8, layers: int = 6, ff: int = 512, clip: float = 10.0):
        super().__init__()
        self.d, self.h, self.clip = d, heads, clip
        self.emb_depot = nn.Linear(2, d)
        self.emb_cust = nn.Linear(3, d)
        self.encoder = nn.ModuleList([EncoderLayer(d, heads, ff) for _ in range(layers)])
        self.Wq = nn.Linear(d + 1, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d)

    def encode(self, coords, demand):
        """coords [B, N+1, 2], demand [B, N+1] normalised by capacity."""
        h = torch.cat([self.emb_depot(coords[:, :1]),
                       self.emb_cust(torch.cat([coords[:, 1:], demand[:, 1:, None]], -1))], 1)
        for layer in self.encoder:
            h = layer(h)
        B, N1, _ = h.shape
        dk = self.d // self.h
        K = self.Wk(h).view(B, N1, self.h, dk).transpose(1, 2)    # [B, H, N+1, dk]
        V = self.Wv(h).view(B, N1, self.h, dk).transpose(1, 2)
        return h, K, V

    def logits(self, h, K, V, last, load, mask):
        """last [B, P] node index, load [B, P] remaining (normalised), mask [B, P, N+1] True = forbidden."""
        B, P = last.shape
        dk = self.d // self.h
        h_last = h.gather(1, last[..., None].expand(B, P, self.d))              # [B, P, d]
        q = self.Wq(torch.cat([h_last, load[..., None]], -1)).view(B, P, self.h, dk).transpose(1, 2)
        score = q @ K.transpose(-1, -2) / math.sqrt(dk)                        # [B, H, P, N+1]
        score = score.masked_fill(mask[:, None], float("-inf"))
        glimpse = (torch.softmax(score, -1) @ V).transpose(1, 2).reshape(B, P, self.d)
        g = self.Wo(glimpse)
        compat = g @ h.transpose(1, 2) / math.sqrt(self.d)                     # [B, P, N+1]
        return (self.clip * torch.tanh(compat)).masked_fill(mask, float("-inf"))


# ----------------------------------------------------------------------------
# Environment (batched, POMO)
# ----------------------------------------------------------------------------
@dataclass
class Rollout:
    length: torch.Tensor      # [B, P]
    logp: torch.Tensor        # [B, P]  sum of log-probabilities of sampled actions
    actions: torch.Tensor     # [B, P, T]


def rollout(model: AttentionPolicy, coords: torch.Tensor, demand: torch.Tensor, greedy: bool = False,
            n_starts: int | None = None) -> Rollout:
    """coords [B, N+1, 2]; demand [B, N+1] already divided by the capacity."""
    B, N1, _ = coords.shape
    n = N1 - 1
    P = n if n_starts is None else min(n_starts, n)
    dev = coords.device
    h, K, V = model.encode(coords, demand)

    bidx = torch.arange(B, device=dev)[:, None].expand(B, P)
    pidx = torch.arange(P, device=dev)[None].expand(B, P)
    cur = torch.zeros(B, P, dtype=torch.long, device=dev)
    load = torch.ones(B, P, device=dev)
    visited = torch.zeros(B, P, N1, dtype=torch.bool, device=dev)
    length = torch.zeros(B, P, device=dev)
    logp = torch.zeros(B, P, device=dev)
    dem = demand[:, None, :].expand(B, P, N1)
    actions = []

    def move(nxt):
        nonlocal cur, load, length
        a = coords[bidx, cur]                     # [B, P, 2]
        b = coords[bidx, nxt]
        length = length + (a - b).norm(dim=-1)
        is_depot = nxt == 0
        q = dem.gather(2, nxt[..., None]).squeeze(-1)
        load = torch.where(is_depot, torch.ones_like(load), load - q)
        visited[bidx, pidx, nxt] = True
        visited[..., 0] = False
        cur = nxt
        actions.append(nxt)

    # POMO: rollout p starts at customer p+1 (forced, not part of the policy gradient)
    move(pidx + 1)
    for _ in range(2 * n + 1):
        done = visited[..., 1:].all(-1)                                  # [B, P]
        if bool(done.all()):
            break
        mask = visited | (dem > load[..., None] + 1e-6)
        mask[..., 0] = cur == 0                                         # no depot -> depot
        mask[..., 0] = mask[..., 0] & ~done                             # finished rollouts may idle at depot
        logits = model.logits(h, K, V, cur, load, mask)
        logprob = torch.log_softmax(logits, -1)
        if greedy:
            nxt = logprob.argmax(-1)
        else:
            nxt = torch.distributions.Categorical(logits=logprob).sample()
        nxt = torch.where(done, torch.zeros_like(nxt), nxt)
        step_lp = logprob.gather(2, nxt[..., None]).squeeze(-1)
        logp = logp + torch.where(done, torch.zeros_like(step_lp), step_lp)
        move(nxt)
    # return to depot
    length = length + (coords[bidx, cur] - coords[:, :1].expand(B, P, 2)).norm(dim=-1)
    return Rollout(length, logp, torch.stack(actions, -1))


def augment8(coords: torch.Tensor) -> torch.Tensor:
    x, y = coords[..., :1], coords[..., 1:]
    variants = [(x, y), (1 - x, y), (x, 1 - y), (1 - x, 1 - y), (y, x), (1 - y, x), (y, 1 - x), (1 - y, 1 - x)]
    return torch.cat([torch.cat(v, -1) for v in variants], 0)


@torch.no_grad()
def solve(model: AttentionPolicy, coords: np.ndarray, demand: np.ndarray, capacity: int,
          augment: bool = True, device: str = "cpu") -> tuple[np.ndarray, list]:
    """Greedy POMO inference with x8 augmentation. Returns (lengths [B], routes per instance)."""
    model.eval()
    c = torch.as_tensor(coords, dtype=torch.float32, device=device)
    q = torch.as_tensor(demand, dtype=torch.float32, device=device) / capacity
    B = c.shape[0]
    if augment:
        c = augment8(c)
        q = q.repeat(8, 1)
    r = rollout(model, c, q, greedy=True)
    L = r.length.view(-1 if not augment else 8, B, r.length.shape[1]) if augment else r.length[None]
    flat = L.permute(1, 0, 2).reshape(B, -1)
    best = flat.argmin(-1)
    lengths = flat.gather(1, best[:, None]).squeeze(1).cpu().numpy()
    acts = r.actions.view(8 if augment else 1, B, *r.actions.shape[1:]).permute(1, 0, 2, 3)
    acts = acts.reshape(B, -1, acts.shape[-1])
    routes = []
    for b in range(B):
        seq = acts[b, best[b]].tolist()
        rts, curr = [], []
        for node in seq:
            if node == 0:
                if curr:
                    rts.append(curr); curr = []
            else:
                curr.append(node)
        if curr:
            rts.append(curr)
        routes.append(rts)
    return lengths, routes


def reinforce_loss(r: Rollout) -> torch.Tensor:
    """POMO shared baseline: advantage = -(L - mean_p L)."""
    adv = -(r.length - r.length.mean(1, keepdim=True))
    return -(adv.detach() * r.logp).mean()
