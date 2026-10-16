import json, numpy as np
from analysis import *
Dm, Da, S = build()
names = [SUBNAME[s] for s in SUBS]
N = {}
def mean(a, n, mode="sp", f="rmse"): return S[f"{mode}|{f}|{a}|{n}"]["mean"]
def vals(a, n, mode="sp", f="rmse"): return np.array(S[f"{mode}|{f}|{a}|{n}"]["vals"])
# win counts (mean RMSE) among FNN/ENN/MRENN
for mode in ("sp", "fr"):
    w = {a: 0 for a in A_MAIN}
    for n in names: w[min(A_MAIN, key=lambda a: mean(a, n, mode))] += 1
    N[f"wins_{mode}"] = w
# matched wins among FNN-m, ENN-m, MRENN
w = {a: 0 for a in ["FNN-m", "ENN-m", "MRENN"]}
for n in names: w[min(w, key=lambda a: mean(a, n))] += 1
N["wins_matched"] = w
# paired geometric-mean ratio MRENN / X with 95% CI over 5 seeds
def ratio(a, b, n, mode="sp"):
    r = np.log(vals(a, n, mode) / vals(b, n, mode)); m, s = r.mean(), r.std(ddof=1)
    h = 2.776 * s / np.sqrt(len(r)); return float(np.exp(m)), float(np.exp(m - h)), float(np.exp(m + h))
N["ratio_fnn"] = {n: ratio("MRENN", "FNN", n) for n in names}
N["ratio_fnnm"] = {n: ratio("MRENN", "FNN-m", n) for n in names}
N["ratio_enn"] = {n: ratio("MRENN", "ENN", n) for n in names}
N["ratio_fnn_fr"] = {n: ratio("MRENN", "FNN", n, "fr") for n in names}
# tests across the 7 sub-problems (on mean RMSE)
def test(a, b, mode="sp"):
    d = [mean(a, n, mode) - mean(b, n, mode) for n in names]; return wilcoxon_exact(d)
N["wilcoxon"] = {f"{a}_vs_{b}_{m}": test(a, b, m) for a, b in [("MRENN", "FNN"), ("MRENN", "ENN"), ("MRENN", "FNN-m"), ("MRENN", "ENN-m"), ("MRENN", "ENN-BPTT"), ("MRENN-noTheta", "MRENN")] for m in ("sp", "fr")}
M = np.array([[mean(a, n) for a in A_MAIN] for n in names]); chi, mr = friedman(M)
N["friedman_sp"] = dict(chi2=chi, ranks=dict(zip(A_MAIN, mr.tolist())), p=chi2_sf(chi, 2))
M = np.array([[mean(a, n, "fr") for a in A_MAIN] for n in names]); chi, mr = friedman(M)
N["friedman_fr"] = dict(chi2=chi, ranks=dict(zip(A_MAIN, mr.tolist())), p=chi2_sf(chi, 2))
A4 = ["FNN-m", "ENN-m", "MRENN"]; M = np.array([[mean(a, n) for a in A4] for n in names]); chi, mr = friedman(M)
N["friedman_matched"] = dict(chi2=chi, ranks=dict(zip(A4, mr.tolist())), p=chi2_sf(chi, 2))
# FR degradation ratios
N["fr_over_sp"] = {a: {n: mean(a, n, "fr") / mean(a, n, "sp") for n in names} for a in A_MAIN}
# noise: mean over seeds of rmse at each level
noise = {}
for a in ["FNN", "ENN", "MRENN", "ENN-BPTT"]:
    for sub in SUBS:
        key, ch = sub; row = []
        for lv in ["0.05", "0.15", "0.3"]:
            row.append(float(np.mean([d["archs"][a]["per_channel"][ch]["noise"][lv]["rmse"] for (k, s), d in Dm.items() if k == key])))
        noise[f"{a}|{SUBNAME[sub]}"] = row
N["noise"] = noise
rel = {}
for a in ["FNN", "ENN", "MRENN"]:
    rel[a] = float(np.mean([noise[f"{a}|{n}"][2] / mean(a, n) for n in names])); 
N["noise_rel30_mean"] = rel
N["noise_rel30_median"] = {a: float(np.median([noise[f"{a}|{n}"][2] / mean(a, n) for n in names])) for a in ["FNN", "ENN", "MRENN"]}
N["noise_abs30_wins"] = {a: 0 for a in A_MAIN}
for n in names: N["noise_abs30_wins"][min(A_MAIN, key=lambda a: noise[f"{a}|{n}"][2])] += 1
# ood
ood = {}
for a in ["FNN", "ENN", "MRENN", "ENN-BPTT"]:
    for sub in SUBS:
        key, ch = sub
        if key == "ex4": continue
        ood[f"{a}|{SUBNAME[sub]}"] = float(np.mean([d["archs"][a]["per_channel"][ch]["ood"]["rmse"] for (k, s), d in Dm.items() if k == key]))
N["ood"] = ood
strat = {}
for a in ["FNN", "ENN", "MRENN"]:
    for sub in SUBS:
        key, ch = sub
        st = [d["archs"][a]["per_channel"][ch]["strat"] for (k, s), d in Dm.items() if k == key]
        strat[f"{a}|{SUBNAME[sub]}"] = {f: float(np.mean([x[f] for x in st])) for f in st[0] if f != "worst_idx"}
N["strat"] = strat
# complexity
cx = {}
for a in A_ALL:
    for sub in SUBS:
        key, ch = sub
        r = [d["archs"][a] for (k, s), d in Dm.items() if k == key]
        cx[f"{a}|{SUBNAME[sub]}"] = dict(params=r[0]["n_params"], macs=r[0]["macs"], state=r[0]["state_floats"], H=r[0]["H"], epochs=r[0]["epochs"],
                                       t_mean=float(np.mean([x["train_time_s"] for x in r])), t_std=float(np.std([x["train_time_s"] for x in r], ddof=1)))
N["cx"] = cx
# MRENN diagnostics
dg = dict(steps=0, descent=0, ratio_max=0.0, proj_wd=0, proj_loop=0, loop_max=0.0, wd_max=0.0)
for (k, s), d in Dm.items():
    r = d["archs"]["MRENN"]; g = r["diag"]; dg["steps"] += g["steps"]; dg["descent"] += g["descent"]; dg["ratio_max"] = max(dg["ratio_max"], g["ratio_max"])
    dg["proj_wd"] += g["proj_wd"]; dg["proj_loop"] += g["proj_loop"]; dg["loop_max"] = max(dg["loop_max"], max(r["loop"])); dg["wd_max"] = max(dg["wd_max"], max(r["wd"]))
N["diag"] = dg
# FR divergence totals
N["fr_div"] = {a: int(sum(S[f"fr|div|{a}|{n}"] for n in names)) for a in A_ALL}
json.dump(N, open(RES + "/numbers.json", "w"), indent=1, default=float)
if __name__ == "__main__":
    for k in ["wins_sp", "wins_fr", "wins_matched", "wilcoxon", "friedman_sp", "friedman_fr", "friedman_matched", "noise_rel30_mean", "noise_rel30_median", "noise_abs30_wins", "diag", "fr_div"]:
        print(k, N[k])
    print("ratio vs FNN", {n: tuple(round(x, 2) for x in v) for n, v in N["ratio_fnn"].items()})
    print("ratio vs FNN-m", {n: tuple(round(x, 2) for x in v) for n, v in N["ratio_fnnm"].items()})
    print("FR/SP", {a: {n[:9]: round(v, 2) for n, v in d.items()} for a, d in N["fr_over_sp"].items()})
