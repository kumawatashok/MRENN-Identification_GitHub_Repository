import json, numpy as np
from analysis import *
S = json.load(open(RES + "/summary.json")); N = json.load(open(RES + "/numbers.json"))
names = [SUBNAME[s] for s in SUBS]
import os; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tables") + os.sep
T = lambda s: s.replace("%", r"\%")


def f(v, d=3):
    if not np.isfinite(v): return r"\textit{div.}"
    return f"{v:.{d}f}"


def pm(a, n, mode="sp", fld="rmse", d=3):
    r = S[f"{mode}|{fld}|{a}|{n}"]; m, s = r["mean"], r["std"]
    return r"\textit{div.}" if not np.isfinite(m) else f"{m:.{d}f}$\\pm${s:.{d}f}"


def best(cols, n, mode="sp"):
    return min(cols, key=lambda a: S[f"{mode}|rmse|{a}|{n}"]["mean"])


def lab(n): return n.replace("Example 5, ch. 1", "Ex. 5, ch. 1").replace("Example 5, ch. 2", "Ex. 5, ch. 2").replace("Example ", "Ex. ")


def wr(name, s): open(OUT + name, "w").write(s)


def tabular_rows(rows): return "\n".join(r"\hline " + " & ".join(r) + r" \\" for r in rows) + "\n\\hline\n"

# --- Table: main accuracy
rows = []
for n in names:
    b = best(["FNN", "ENN", "MRENN"], n); r = [lab(n)]
    for a in ["FNN", "ENN", "MRENN"]:
        c = pm(a, n); r.append(r"\textbf{" + c + "}" if a == b else c); r.append(pm(a, n, fld="mae"))
    rows.append(r)
wr("main.tex", r"""\begin{tabular}{|l|cc|cc|cc|}
\hline
 & \multicolumn{2}{c|}{\textbf{FNN}} & \multicolumn{2}{c|}{\textbf{ENN}} & \multicolumn{2}{c|}{\textbf{MRENN}} \\
\textbf{Benchmark} & RMSE & MAE & RMSE & MAE & RMSE & MAE \\
""" + tabular_rows(rows) + r"\end{tabular}")
# --- free running
rows = []
for n in names:
    b = best(["FNN", "ENN", "MRENN"], n, "fr"); r = [lab(n)]
    for a in ["FNN", "ENN", "MRENN"]:
        c = pm(a, n, "fr"); r.append(r"\textbf{" + c + "}" if a == b else c)
        r.append(f"{S[f'fr|rmse|{a}|{n}']['mean'] / S[f'sp|rmse|{a}|{n}']['mean']:.2f}")
    rows.append(r)
wr("free.tex", r"""\begin{tabular}{|l|cc|cc|cc|}
\hline
 & \multicolumn{2}{c|}{\textbf{FNN}} & \multicolumn{2}{c|}{\textbf{ENN}} & \multicolumn{2}{c|}{\textbf{MRENN}} \\
\textbf{Benchmark} & RMSE & $\times$SP & RMSE & $\times$SP & RMSE & $\times$SP \\
""" + tabular_rows(rows) + r"\end{tabular}")
# --- matched + BPTT
rows = []
for n in names:
    cols = ["FNN-m", "ENN-m", "MRENN", "ENN-BPTT"]; b = best(cols, n); r = [lab(n)]
    for a in cols:
        c = pm(a, n); r.append(r"\textbf{" + c + "}" if a == b else c)
    q = N["ratio_fnnm"][n]; r.append(f"{q[0]:.2f} [{q[1]:.2f}, {q[2]:.2f}]")
    rows.append(r)
wr("matched.tex", r"""\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Benchmark} & \textbf{FNN (matched)} & \textbf{ENN (matched)} & \textbf{MRENN} & \textbf{ENN-BPTT} & \textbf{MRENN / FNN (matched)} \\
""" + tabular_rows(rows) + r"\end{tabular}")
# --- ablation
rows = []
cols = ["MRENN", "MRENN-noTheta", "MRENN-noC2", "MRENN-noWd", "MRENN-lag2", "MRENN-fixedLR"]
for n in names:
    b = best(cols, n); r = [lab(n)]
    for a in cols:
        c = pm(a, n); r.append(r"\textbf{" + c + "}" if a == b else c)
    rows.append(r)
wr("ablation.tex", r"""\begin{tabular}{|l|c|c|c|c|c|c|}
\hline
\textbf{Benchmark} & \textbf{Full} & \textbf{No $\boldsymbol{\Theta}$} & \textbf{No $c2$} & \textbf{No $W_d$} & \textbf{Lag 2} & \textbf{Fixed $\eta$} \\
""" + tabular_rows(rows) + r"\end{tabular}")
# --- complexity (params, MACs)
rows = []
for n in names:
    c = {a: N["cx"][f"{a}|{n}"] for a in ["FNN", "ENN", "MRENN"]}
    sav = 100 * (1 - c["MRENN"]["params"] / c["FNN"]["params"])
    rows.append([lab(n), str(c["FNN"]["params"]), str(c["ENN"]["params"]), str(c["MRENN"]["params"]), f"{sav:.0f}\\%",
                 str(c["FNN"]["macs"]), str(c["ENN"]["macs"]), str(c["MRENN"]["macs"])])
wr("params.tex", r"""\begin{tabular}{|l|cccc|ccc|}
\hline
 & \multicolumn{4}{c|}{\textbf{Trainable parameters}} & \multicolumn{3}{c|}{\textbf{MACs per step}} \\
\textbf{Benchmark} & FNN & ENN & MRENN & Saving & FNN & ENN & MRENN \\
""" + tabular_rows(rows) + r"\end{tabular}")
# --- timing
Dm = load("main")
def tm(a, key):
    r = [d["archs"][a] for (k, s), d in Dm.items() if k == key]
    return np.mean([x["train_time_s"] for x in r]), np.std([x["train_time_s"] for x in r], ddof=1), np.mean([x["infer_us_per_step"] for x in r])
timing = json.load(open(RES + "/timing.json")) if os.path.exists(RES + "/timing.json") else {}
rows = []
for a, nm in [("FNN", "FNN"), ("FNN-m", "FNN (matched)"), ("ENN", "ENN"), ("ENN-m", "ENN (matched)"), ("ENN-BPTT", "ENN-BPTT"), ("MRENN", "MRENN")]:
    t1 = tm(a, "ex1"); t5 = tm(a, "ex5")
    c1 = N["cx"][f"{a}|Example 1"]; c5 = N["cx"][f"{a}|Example 5, ch. 1"]
    mem1 = (c1["params"] + c1["state"]) * 8
    rows.append([nm, f"{c1['H']}", f"{c1['epochs']}", f"{t1[0]:.1f}$\\pm${t1[1]:.1f}", f"{t5[0]:.1f}$\\pm${t5[1]:.1f}", f"{timing.get(a, {}).get('ex1', float('nan')):.1f}", f"{mem1}"])
wr("timing.tex", r"""\begin{tabular}{|l|c|c|c|c|c|c|}
\hline
\textbf{Model} & \textbf{Hidden} & \textbf{Epochs} & \textbf{Train, Ex. 1 (s)} & \textbf{Train, Ex. 5 (s)} & \textbf{Infer. ($\mu$s/step)} & \textbf{Memory (B)} \\
""" + tabular_rows(rows) + r"\end{tabular}")
# --- robustness: noise 30 % and out-of-range
rows = []
for n in names:
    r = [lab(n)]
    for a in ["FNN", "ENN", "MRENN"]: r.append(f"{N['noise'][f'{a}|{n}'][2]:.3f}")
    for a in ["FNN", "ENN", "MRENN"]: r.append(f"{N['ood'][f'{a}|{n}']:.3f}" if f"{a}|{n}" in N["ood"] else "n/a")
    rows.append(r)
wr("robust.tex", r"""\begin{tabular}{|l|ccc|ccc|}
\hline
 & \multicolumn{3}{c|}{\textbf{RMSE at 30 \% noise}} & \multicolumn{3}{c|}{\textbf{RMSE, input $\times 1.5$}} \\
\textbf{Benchmark} & FNN & ENN & MRENN & FNN & ENN & MRENN \\
""" + tabular_rows(rows) + r"\end{tabular}")
# --- stratified
rows = []
for n in names:
    r = [lab(n)]
    for a in ["FNN", "MRENN"]:
        s = N["strat"][f"{a}|{n}"]; r += [f"{s['low']:.3f}", f"{s['mid']:.3f}", f"{s['high']:.3f}"]
    rows.append(r)
wr("strat.tex", r"""\begin{tabular}{|l|ccc|ccc|}
\hline
 & \multicolumn{3}{c|}{\textbf{FNN}} & \multicolumn{3}{c|}{\textbf{MRENN}} \\
\textbf{Benchmark} & Low & Mid & High & Low & Mid & High \\
""" + tabular_rows(rows) + r"\end{tabular}")
# --- statistics
fs, ff, fm = N["friedman_sp"], N["friedman_fr"], N["friedman_matched"]
rows = [["Series-parallel", "4 / 0 / 3", f"{fs['ranks']['FNN']:.2f} / {fs['ranks']['ENN']:.2f} / {fs['ranks']['MRENN']:.2f}", f"{fs['chi2']:.2f}", f"{fs['p']:.2f}"],
        ["Free-running", "4 / 1 / 2", f"{ff['ranks']['FNN']:.2f} / {ff['ranks']['ENN']:.2f} / {ff['ranks']['MRENN']:.2f}", f"{ff['chi2']:.2f}", f"{ff['p']:.2f}"],
        ["Matched protocol", "2 / 0 / 5", f"{fm['ranks']['FNN-m']:.2f} / {fm['ranks']['ENN-m']:.2f} / {fm['ranks']['MRENN']:.2f}", f"{fm['chi2']:.2f}", f"{fm['p']:.2f}"]]
wr("stats.tex", r"""\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Evaluation} & \textbf{Wins} & \textbf{Mean ranks} & $\boldsymbol{\chi^2}$ & $\boldsymbol{p}$ \\
""" + tabular_rows(rows) + r"\end{tabular}")
print("ok")
