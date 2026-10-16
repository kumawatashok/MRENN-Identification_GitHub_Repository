"""Hyper-parameter selection on a held-out slice of the TRAINING sequence only
(first 75 % train, last 25 % hold-out, seed 100).  The validation sequences used in
the paper's tables never enter this procedure."""
import sys, os, json, itertools, time
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from core import *

SETS = [("ex1", 0), ("ex2", 0), ("ex3", 0), ("ex4", 0), ("ex5", 0), ("ex5", 1), ("ex6", 0)]


def sel_path(arch, name, key, ch):
    return os.path.join(HERE, "..", "results", "sel", f"{arch}__{name}__{key}{ch}.json")


def grid():
    D = DEFAULT; g = []
    for H in (10, 24, 48):
        for lr in (0.005, 0.02, 0.05): g.append(("FNN", f"H{H}_lr{lr}", dict(D["FNN"], H=H, lr=lr)))
    for lr in (0.001, 0.005, 0.02):
        for mo in (0.0, 0.9): g.append(("ENN", f"lr{lr}_mom{mo}", dict(D["ENN"], lr=lr, mom=mo)))
    for lr in (0.003, 0.01): g.append(("ENN-BPTT", f"lr{lr}", dict(D["ENN-BPTT"], lr=lr)))
    for H in (10, 20):
        for r in (0.1, 0.25, 0.5, 1.0): g.append(("MRENN", f"H{H}_rho{r}", dict(D["MRENN"], H=H, rho=r)))
    return g


def sel_task(arch, name, key, ch):
    cfg = [c for a, n, c in grid() if a == arch and n == name][0]
    r = one((arch, cfg, key, ch))
    os.makedirs(os.path.dirname(sel_path(arch, name, key, ch)), exist_ok=True)
    json.dump(dict(r=r), open(sel_path(arch, name, key, ch), "w"))


def tasks():
    return [("sel", a, n, k, c) for a, n, cfg in grid() for k, c in SETS]


def one(args):
    arch, cfg, key, ch = args
    u, y, _, _ = prepare(key, 100)
    K = len(u); kc = int(.75 * K)
    kind = cfg["kind"]
    X = build_X(key, kind, y, u, K); Yall = y[1:K + 1]
    Y = Yall[:, ch] if PLANTS[key]["mimo"] else Yall
    m, curve, dt = train(arch, cfg, X[:kc], Y[:kc], 100)
    pred = np.array([m.step(x) for x in X[kc:]]) if not isinstance(m, FNN) else m.predict(X[kc:])
    ho = Y[kc:]
    r = float(np.sqrt(np.mean((pred - ho) ** 2))) / float(ho.std())
    return r if np.isfinite(r) else 1e3


def summarize():
    out = {}
    for a, n, cfg in grid():
        rs = []
        for k, c in SETS:
            p = sel_path(a, n, k, c)
            if not os.path.exists(p): rs = None; break
            rs.append(json.load(open(p))["r"])
        if rs is None: continue
        out.setdefault(a, []).append(dict(name=n, cfg=cfg, per_set=rs, score=float(np.mean(np.minimum(rs, 10.0)))))
    return out
