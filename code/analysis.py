import json, os, numpy as np
from aggregate import *

A_MAIN = ["FNN", "ENN", "MRENN"]
A_ALL = ["FNN", "ENN", "MRENN", "ENN-BPTT", "FNN-m", "ENN-m"]
A_ABL = ["MRENN-noTheta", "MRENN-noC2", "MRENN-noWd", "MRENN-lag2", "MRENN-fixedLR"]
T975 = {4: 2.776}


def ms(v): return float(np.mean(v)), float(np.std(v, ddof=1))


def wilcoxon_exact(d):
    """two-sided exact Wilcoxon signed-rank on differences d (zeros dropped)."""
    d = np.array([x for x in d if x != 0.0]); n = len(d)
    if n == 0: return 1.0
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    # average ranks for ties
    a = np.abs(d)
    for val in np.unique(a):
        m = a == val; r[m] = r[m].mean()
    wplus = r[d > 0].sum(); tot = r.sum()
    # exact distribution by enumeration (n<=10 fine)
    import itertools
    cnt = 0; N = 0; obs = min(wplus, tot - wplus)
    for signs in itertools.product([0, 1], repeat=n):
        w = sum(ri for ri, s in zip(r, signs) if s); N += 1
        if min(w, tot - w) <= obs + 1e-9: cnt += 1
    return cnt / N


def friedman(M):
    """M: n_blocks x k_treatments (lower is better). returns chi2, mean ranks"""
    n, k = M.shape
    R = np.zeros_like(M)
    for i in range(n):
        o = np.argsort(M[i]); rk = np.empty(k); rk[o] = np.arange(1, k + 1)
        for val in np.unique(M[i]):
            m = M[i] == val; rk[m] = rk[m].mean()
        R[i] = rk
    mr = R.mean(0)
    chi2 = 12 * n / (k * (k + 1)) * np.sum((mr - (k + 1) / 2) ** 2)
    return float(chi2), mr


def chi2_sf(x, df):
    # survival function for small integer df via series
    import math
    if df == 2: return math.exp(-x / 2)
    if df == 3:
        return math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    raise NotImplementedError


def build():
    Dm, Da = load("main"), load("abl")
    S = {}
    # --- accuracy tables (mean/std over seeds), series-parallel and free-running
    for mode in ("sp", "fr"):
        for field in ("rmse", "mae", "nrmse", "mape_f"):
            for a in A_ALL:
                for sub in SUBS:
                    v = collect(Dm, a, sub, field, mode)
                    S[f"{mode}|{field}|{a}|{SUBNAME[sub]}"] = dict(mean=float(np.nanmean(v)), std=float(np.nanstd(v, ddof=1)), vals=v.tolist(),
                                                              median=float(np.nanmedian(v)))
    for a in A_ABL:
        for sub in SUBS:
            for mode in ("sp", "fr"):
                v = collect(Da, a, sub, "rmse", mode)
                S[f"{mode}|rmse|{a}|{SUBNAME[sub]}"] = dict(mean=float(np.nanmean(v)), std=float(np.nanstd(v, ddof=1)), vals=v.tolist(), median=float(np.nanmedian(v)))
    # divergence counts in free run
    for a in A_ALL:
        for sub in SUBS:
            key, ch = sub
            c = sum(int(d["archs"][a]["per_channel"][ch]["fr"]["diverged"]) for (k, s), d in Dm.items() if k == key)
            S[f"fr|div|{a}|{SUBNAME[sub]}"] = c
    json.dump(S, open(os.path.join(RES, "summary.json"), "w"), default=float)
    return Dm, Da, S


if __name__ == "__main__":
    Dm, Da, S = build()
    names = [SUBNAME[s] for s in SUBS]
    print("SP RMSE mean±std (n=5)")
    for n in names:
        print(f"{n:18s}", "  ".join(f"{a}:{S[f'sp|rmse|{a}|{n}']['mean']:.3f}±{S[f'sp|rmse|{a}|{n}']['std']:.3f}" for a in A_ALL))
    print("ablation SP RMSE")
    for n in names:
        print(f"{n:18s} MRENN:{S[f'sp|rmse|MRENN|{n}']['mean']:.3f}", "  ".join(f"{a[6:]}:{S[f'sp|rmse|{a}|{n}']['mean']:.3f}±{S[f'sp|rmse|{a}|{n}']['std']:.3f}" for a in A_ABL))
    print("FR diverged counts")
    for n in names:
        print(n, {a: S[f'fr|div|{a}|{n}'] for a in A_ALL})
    # stats on nrmse means
    M = np.array([[S[f"sp|nrmse|{a}|{n}"]["mean"] for a in A_MAIN] for n in names])
    chi, mr = friedman(M); print("Friedman main", chi, mr, "p", chi2_sf(chi, 2))
    print("Wilcoxon MRENN vs FNN p", wilcoxon_exact(M[:, 2] - M[:, 0]), "MRENN vs ENN", wilcoxon_exact(M[:, 2] - M[:, 1]))
    d = Dm[("ex3", 0)]["archs"]["MRENN"]; print(d["diag"], d["wd"], d["loop"])
    for k, s in sorted(Dm):
        dd = Dm[(k, s)]["archs"]["MRENN"]["diag"]
        print(k, s, dd["descent"] / dd["steps"], round(dd["ratio_max"], 3), round(dd["ratio_mean"], 3), dd["proj_wd"], dd["proj_loop"], [round(x, 3) for x in Dm[(k, s)]["archs"]["MRENN"]["loop"]])
