"""Run all experiments of the revised study for one (benchmark, seed) job.

usage:  python run_experiments.py <ex1..ex6> <seed> [group]
group = main (FNN, ENN, MRENN, ENN-BPTT, FNN-m, ENN-m)  |  abl (MRENN ablations)
Results -> ../results/raw/<key>_<seed>_<group>.json (+ .npz with seed-0 traces)
"""
import sys, os, json, time
import numpy as np
from core import *
np.seterr(all="ignore")
from plants import simulate

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

GROUPS = {"main": ["FNN", "ENN", "MRENN", "ENN-BPTT", "FNN-m", "ENN-m"],
          "abl": ["MRENN-noTheta", "MRENN-noC2", "MRENN-noWd", "MRENN-lag2", "MRENN-fixedLR"]}
NOISE = (0.05, 0.15, 0.30)
NOISE_ARCHS = ("FNN", "ENN", "MRENN", "ENN-BPTT")
OOD_ARCHS = ("FNN", "ENN", "MRENN", "ENN-BPTT")


def load_cfgs():
    cfg = {k: dict(v) for k, v in DEFAULT.items()}
    p = os.path.join(RES, "selected.json")
    if os.path.exists(p):
        sel = json.load(open(p))
        for k, v in sel.items(): cfg[k].update(v)
    # matched-budget models share the MRENN rate / width (same rule, same width, same epochs)
    cfg["FNN-m"].update(H=cfg["MRENN"]["H"], rho=cfg["MRENN"]["rho"], eps=cfg["MRENN"]["eps"], epochs=cfg["MRENN"]["epochs"])
    cfg["ENN-m"].update(H=cfg["MRENN"]["H"], rho=cfg["MRENN"]["rho"], eps=cfg["MRENN"]["eps"], epochs=cfg["MRENN"]["epochs"])
    cfg["MRENN-noTheta"] = dict(cfg["MRENN"], flags=dict(use_theta=False))
    cfg["MRENN-noC2"] = dict(cfg["MRENN"], flags=dict(use_c2=False))
    cfg["MRENN-noWd"] = dict(cfg["MRENN"], flags=dict(use_wd=False))
    cfg["MRENN-lag2"] = dict(cfg["MRENN"], kind="min2")
    cfg["MRENN-fixedLR"] = dict(cfg["MRENN"], eta_fixed=0.02)
    return cfg


def main(key, seed, group):
    cfgs = load_cfgs()
    spec = PLANTS[key]; mimo = spec["mimo"]
    u_tr, y_tr, u_va, y_va = prepare(key, seed)
    K = len(u_tr); Kv = len(u_va)
    chans = [0, 1] if mimo else [None]
    Ytr = lambda c: (y_tr[1:K + 1, c] if mimo else y_tr[1:K + 1])
    Yva = lambda c: (y_va[1:Kv + 1, c] if mimo else y_va[1:Kv + 1])
    u_oo = 1.5 * u_va; y_oo = simulate(key, u_oo)
    out = dict(key=key, seed=seed, archs={}); traces = {}
    rngn = np.random.default_rng(777 + seed)
    for arch in GROUPS[group]:
        cfg = cfgs[arch]; kind = cfg["kind"]
        Xtr = build_X(key, kind, y_tr, u_tr, K); Xva = build_X(key, kind, y_va, u_va, Kv)
        models, curves, ttrain = [], [], 0.0
        for c in chans:
            m, curve, dt = train(arch, cfg, Xtr, Ytr(c), seed, diag=(arch == "MRENN"))
            models.append(m); curves.append(curve); ttrain += dt
        R = dict(n_params=int(models[0].n_params), macs=int(models[0].macs()), state_floats=int(models[0].state_floats),
                 H=cfg["H"], epochs=cfg["epochs"], train_time_s=ttrain / len(chans), per_channel=[])
        t0 = time.perf_counter()
        sp = [sp_predict(m, Xva) for m in models]
        R["infer_us_per_step"] = (time.perf_counter() - t0) / (Kv * len(chans)) * 1e6
        fr, dv = free_run(models, key, kind, u_va)
        for i, c in enumerate(chans):
            ytrue = Yva(c); floor = 0.05 * float(np.abs(ytrue).max())
            P = dict(sp=metrics(sp[i], ytrue, floor))
            frc = fr[:, c] if mimo else fr
            P["fr"] = metrics(frc, ytrue, floor); P["fr"]["diverged"] = bool(dv)
            P["train_final"] = curves[i][-1]
            P["curve"] = curves[i] if arch in ("FNN", "ENN", "MRENN", "ENN-BPTT") else curves[i][-1:]
            # stratified statistics (series-parallel)
            e = sp[i] - ytrue; a = np.abs(ytrue); q1, q2 = np.quantile(a, [1 / 3, 2 / 3])
            P["strat"] = dict(low=float(np.sqrt(np.mean(e[a <= q1] ** 2))), mid=float(np.sqrt(np.mean(e[(a > q1) & (a <= q2)] ** 2))),
                              high=float(np.sqrt(np.mean(e[a > q2] ** 2))))
            if not mimo:
                P["strat"].update(seg1=float(np.sqrt(np.mean(e[:500] ** 2))), seg2=float(np.sqrt(np.mean(e[500:] ** 2))))
            P["strat"]["worst_idx"] = int(np.argmax(np.abs(e)))
            if arch in OOD_ARCHS:
                Xo = build_X(key, kind, y_oo, u_oo, Kv)
                po = sp_predict(models[i], Xo)
                yo = y_oo[1:Kv + 1, c] if mimo else y_oo[1:Kv + 1]
                P["ood"] = metrics(po, yo, 0.05 * float(np.abs(yo).max()))
            if arch in NOISE_ARCHS:
                P["noise"] = {}
                for lv in NOISE:
                    yn = y_va + lv * y_va.std(0) * rngn.standard_normal(y_va.shape)
                    un = u_va + lv * u_va.std(0) * rngn.standard_normal(u_va.shape)
                    pn = sp_predict(models[i], build_X(key, kind, yn, un, Kv))
                    P["noise"][str(lv)] = metrics(pn, ytrue, floor)
            R["per_channel"].append(P)
            if seed == 0:
                traces[f"{arch}|{i}|sp"] = sp[i]; traces[f"{arch}|{i}|fr"] = frc
        if arch == "MRENN":
            st = models[0].stats; R["diag"] = dict(st, steps=int(st["steps"]), descent=int(st["descent"]), ratio_max=float(st["ratio_max"]),
                                                  ratio_mean=float(st["ratio_sum"] / max(st["steps"], 1)))
            R["wd"] = [float(np.abs(m.Wd).max()) for m in models]; R["loop"] = [float(abs(m.Wf + m.Woc)) for m in models]
        out["archs"][arch] = R
        print(key, seed, arch, [round(p["sp"]["rmse"], 4) for p in R["per_channel"]], [round(p["fr"]["rmse"], 4) for p in R["per_channel"]],
              round(R["train_time_s"], 1), flush=True)
    os.makedirs(os.path.join(RES, "raw"), exist_ok=True)
    json.dump(out, open(os.path.join(RES, "raw", f"{key}_{seed}_{group}.json"), "w"), default=float)
    if seed == 0:
        traces["y_true"] = y_va[1:Kv + 1]; traces["u_va"] = u_va; traces["y_tr"] = y_tr; traces["u_tr"] = u_tr
        np.savez_compressed(os.path.join(RES, "raw", f"{key}_{seed}_{group}.npz"), **traces)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "main")
