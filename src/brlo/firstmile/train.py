"""Train / evaluate the POMO attention policy for first-mile collection.

Examples
--------
Smoke test (CPU, < 1 min)::

    python -m brlo.firstmile.train --n 20 --epochs 2 --steps 20 --batch 32 --layers 3

Paper setting (GPU recommended)::

    python -m brlo.firstmile.train --n 50 --dist clustered --epochs 200 --steps 500 --batch 64

Evaluate a checkpoint against Clarke-Wright + local search::

    python -m brlo.firstmile.train --eval runs/pomo_n50_clustered/best.pt --n 50 --dist clustered
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from . import instances as gen
from .heuristics import dist_matrix, solve_classical, total_length
from .neural import AttentionPolicy, reinforce_loss, rollout, solve


def make_batch(dist: str, batch: int, n: int, seed=None):
    return (gen.clustered if dist == "clustered" else gen.uniform)(batch, n, seed)


def to_tensors(b, device):
    c = torch.as_tensor(b.coords, dtype=torch.float32, device=device)
    q = torch.as_tensor(b.demand, dtype=torch.float32, device=device) / b.capacity
    return c, q


def classical_reference(val, method="cw+ls"):
    out, t0 = [], time.perf_counter()
    for i in range(len(val.coords)):
        r = solve_classical(val.coords[i], val.demand[i], val.capacity, method)
        out.append(total_length(r, dist_matrix(val.coords[i])))
    return np.array(out), (time.perf_counter() - t0) / len(out)


def train(args) -> str:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    run = os.path.join(args.out, f"pomo_n{args.n}_{args.dist}")
    os.makedirs(run, exist_ok=True)
    with open(os.path.join(run, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)
    model = AttentionPolicy(d=args.d, heads=args.heads, layers=args.layers, ff=4 * args.d).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(0.9 * args.epochs)], gamma=0.1)
    val = make_batch(args.dist, args.val_size, args.n, seed=12345)
    ref, ref_t = classical_reference(val)
    print(f"[val] CW+LS mean length {ref.mean():.4f}  ({1000 * ref_t:.1f} ms/instance)")
    best = np.inf
    log = open(os.path.join(run, "log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["epoch", "loss", "train_len", "val_len", "val_gap_vs_cwls_pct", "val_ms_per_inst", "seconds"])
    t_start = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        losses, lens = [], []
        for _ in range(args.steps):
            c, q = to_tensors(make_batch(args.dist, args.batch, args.n), dev)
            r = rollout(model, c, q, greedy=False)
            loss = reinforce_loss(r)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            lens.append(r.length.min(1).values.mean().item())
        sched.step()
        t0 = time.perf_counter()
        vl, _ = solve(model, val.coords, val.demand, val.capacity, augment=True, device=args.device)
        ms = 1000 * (time.perf_counter() - t0) / len(vl)
        gap = 100 * (vl.mean() / ref.mean() - 1)
        w.writerow([ep, np.mean(losses), np.mean(lens), vl.mean(), gap, ms, time.perf_counter() - t_start])
        log.flush()
        print(f"ep {ep:4d} loss {np.mean(losses):+.4f} train {np.mean(lens):.4f} val {vl.mean():.4f} "
              f"({gap:+.2f}% vs CW+LS, {ms:.1f} ms/inst)")
        torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(run, "last.pt"))
        if vl.mean() < best:
            best = vl.mean()
            torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(run, "best.pt"))
    log.close()
    return run


def evaluate_ckpt(path, n, dist, size=200, device="cpu"):
    ck = torch.load(path, map_location=device)
    a = ck["args"]
    model = AttentionPolicy(d=a["d"], heads=a["heads"], layers=a["layers"], ff=4 * a["d"]).to(device)
    model.load_state_dict(ck["model"])
    test = make_batch(dist, size, n, seed=999)
    rows = {}
    for m in ("nn", "cw", "cw+ls"):
        L, t = classical_reference(test, m)
        rows[m] = (L.mean(), 1000 * t)
    for aug in (False, True):
        t0 = time.perf_counter()
        L, routes = solve(model, test.coords, test.demand, test.capacity, augment=aug, device=device)
        rows[f"pomo{'-x8' if aug else ''}"] = (L.mean(), 1000 * (time.perf_counter() - t0) / size)
    base = rows["cw+ls"][0]
    print(f"{'method':10s} {'length':>8s} {'gap vs CW+LS':>13s} {'ms/inst':>8s}")
    for k, (L, ms) in rows.items():
        print(f"{k:10s} {L:8.4f} {100 * (L / base - 1):+12.2f}% {ms:8.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--dist", choices=["uniform", "clustered"], default="clustered")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--val-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--eval", default=None, help="checkpoint to evaluate instead of training")
    args = ap.parse_args()
    if args.eval:
        evaluate_ckpt(args.eval, args.n, args.dist, device=args.device)
    else:
        train(args)


if __name__ == "__main__":
    main()
