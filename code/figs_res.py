import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from aggregate import *
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "grid.linestyle": ":"})
import os; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures") + os.sep
C = {"plant": "k", "FNN": "#1f6bff", "ENN": "#12a02a", "MRENN": "#e3202b", "ENN-BPTT": "#ff8a00", "FNN-m": "#00b8d9", "ENN-m": "#9b30ff"}
LS = {"FNN": "--", "ENN": ":", "MRENN": "-"}
NAMES = [SUBNAME[s] for s in SUBS]
SHORT = ["Ex. 1", "Ex. 2", "Ex. 3", "Ex. 4", "Ex. 5-1", "Ex. 5-2", "Ex. 6"]
Dm, Da = load("main"), load("abl")
S = json.load(open(RES + "/summary.json"))


def rm(a, n, mode="sp"): return S[f"{mode}|rmse|{a}|{n}"]


def fig_convergence():
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.8)); axs = axs.ravel()
    for p, sub in enumerate(SUBS):
        ax = axs[p]; key, ch = sub
        for a in ["FNN", "ENN", "MRENN"]:
            cs = np.array([d["archs"][a]["per_channel"][ch]["curve"] for (k, s), d in sorted(Dm.items()) if k == key])
            ep = np.arange(1, cs.shape[1] + 1); med = np.median(cs, 0)
            ax.fill_between(ep, cs.min(0), cs.max(0), color=C[a], alpha=.22, lw=0)
            ax.plot(ep, med, color=C[a], ls=LS[a], lw=2, label=f"{a}  (final {med[-1]:.3f})")
        ax.set_yscale("log"); ax.set_title(NAMES[p], fontsize=11, fontweight="bold"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8, frameon=False, loc="upper right")
        if p % 4 == 0: ax.set_ylabel("Training RMSE")
    axs[7].axis("off")
    axs[7].text(0, .62, "Median over 5 seeds (line)\nand min–max band (shading).\nTraining RMSE on the clean\ntraining signal; online rules\nreport the running a-priori error.\nEpoch budgets: FNN 800,\nENN 300, MRENN 150.", fontsize=10, va="center")
    fig.tight_layout(); fig.savefig(OUT + "fig07_train_rmse_convergence.png", dpi=300); plt.close(fig)


def fig_val_summary():
    fig, ax = plt.subplots(figsize=(11, 4.8)); w = .26
    for i, a in enumerate(["FNN", "ENN", "MRENN"]):
        m = [rm(a, n)["mean"] for n in NAMES]; s = [rm(a, n)["std"] for n in NAMES]
        ax.bar(np.arange(7) + (i - 1) * w, m, w, yerr=s, color=C[a], capsize=3, label=a, edgecolor="k", lw=.6, error_kw=dict(lw=1.2))
        for x, v in zip(np.arange(7) + (i - 1) * w, m): ax.text(x, v * 1.25, f"{v:.2f}" if v >= .1 else f"{v:.3f}", ha="center", fontsize=7, rotation=90, va="bottom")
    ax.set_yscale("log"); ax.set_ylim(.008, 6); ax.set_xticks(range(7)); ax.set_xticklabels(SHORT); ax.set_ylabel("Validation RMSE (mean ± std, 5 seeds)"); ax.legend(ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(OUT + "fig08_val_rmse_summary.png", dpi=300); plt.close(fig)


def zoom_window(y, L=220):
    best, bi = -1, 0
    for i in range(0, len(y) - L, 10):
        v = np.std(y[i:i + L])
        if v > best: best, bi = v, i
    return bi, bi + L


def fig_examples():
    files = {}
    for key, num in [("ex1", 9), ("ex2", 10), ("ex3", 11), ("ex4", 12), ("ex5", 13), ("ex6", 15)]:
        z = np.load(RES + f"/raw/{key}_0_main.npz")
        chs = [0, 1] if key == "ex5" else [None]
        for i, c in enumerate(chs):
            yt = z["y_true"][:, c] if c is not None else z["y_true"]
            lo, hi = zoom_window(yt)
            fig = plt.figure(figsize=(16, 4.3)); gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.3, 2.2], wspace=.22)
            axs = [fig.add_subplot(gs[j]) for j in range(3)]; t = np.arange(len(yt)) + 1
            lab = NAMES[SUBS.index((key, i if c is not None else 0))]
            rms = {}
            for a in ["FNN", "ENN", "MRENN"]:
                sp = z[f"{a}|{i}|sp"]; fr = z[f"{a}|{i}|fr"]
                rms[a] = (np.sqrt(np.mean((sp - yt) ** 2)), np.sqrt(np.mean((fr - yt) ** 2)))
                for ax, y, sl in ((axs[0], sp, slice(None)), (axs[1], sp, slice(lo, hi)), (axs[2], fr, slice(None))):
                    ax.plot(t[sl], y[sl], color=C[a], ls=LS[a], lw=1.7)
            for ax, sl in ((axs[0], slice(None)), (axs[1], slice(lo, hi)), (axs[2], slice(None))): ax.plot(t[sl], yt[sl], color="k", lw=2.2, zorder=1)
            axs[0].axvspan(lo, hi, color="#cccccc", alpha=.6, zorder=0); axs[1].set_facecolor("#fdfaf0")
            axs[0].set_title(f"(a) {lab}: series-parallel, full window", fontsize=10, fontweight="bold")
            axs[1].set_title(f"(b) zoom, k = {lo}–{hi}", fontsize=10, fontweight="bold")
            axs[2].set_title("(c) free-running, full window", fontsize=10, fontweight="bold")
            for ax in axs: ax.set_xlabel("Time step k")
            axs[0].set_ylabel("Plant / model output")
            h = [Line2D([], [], color="k", lw=2.2, label="plant $y_p(k)$")] + [Line2D([], [], color=C[a], ls=LS[a], lw=1.8, label=f"{a} (RMSE {rms[a][0]:.3f} | {rms[a][1]:.3f})") for a in rms]
            fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(.5, .0))
            fig.subplots_adjust(bottom=.30, top=.9, left=.05, right=.99)
            name = {("ex1", 0): "fig09_example1", ("ex2", 0): "fig10_example2", ("ex3", 0): "fig11_example3", ("ex4", 0): "fig12_example4",
                    ("ex5", 0): "fig13_example5_ch1", ("ex5", 1): "fig14_example5_ch2", ("ex6", 0): "fig15_example6"}[(key, i if c is not None else 0)]
            fig.savefig(OUT + name + ".png", dpi=300); plt.close(fig)


def fig_fr():
    fig, ax = plt.subplots(figsize=(11.5, 4.8)); w = .13
    k = 0
    for a in ["FNN", "ENN", "MRENN"]:
        for mode, hatch in (("sp", ""), ("fr", "///")):
            m = [rm(a, n, mode)["mean"] for n in NAMES]; s = [rm(a, n, mode)["std"] for n in NAMES]
            ax.bar(np.arange(7) + (k - 2.5) * w, m, w, yerr=s, color=C[a], hatch=hatch, edgecolor="k", lw=.6, capsize=2, error_kw=dict(lw=1))
            k += 1
    ax.set_yscale("log"); ax.set_xticks(range(7)); ax.set_xticklabels(SHORT); ax.set_ylabel("Validation RMSE (mean ± std)")
    h = [plt.Rectangle((0, 0), 1, 1, fc=C[a], ec="k") for a in ["FNN", "ENN", "MRENN"]] + [plt.Rectangle((0, 0), 1, 1, fc="w", ec="k"), plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", hatch="///")]
    ax.legend(h, ["FNN", "ENN", "MRENN", "series-parallel", "free-running"], ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.12))
    fig.tight_layout(); fig.savefig(OUT + "fig16_free_running.png", dpi=300); plt.close(fig)


def fig_matched_ablation():
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.9))
    ax = axs[0]; arch = ["FNN-m", "ENN-m", "MRENN", "ENN-BPTT"]; lab = ["FNN (matched)", "ENN (matched)", "MRENN", "ENN-BPTT (tuned)"]; w = .2
    for i, a in enumerate(arch):
        m = [rm(a, n)["mean"] for n in NAMES]; s = [rm(a, n)["std"] for n in NAMES]
        ax.bar(np.arange(7) + (i - 1.5) * w, m, w, yerr=s, color=C[a], edgecolor="k", lw=.6, capsize=2, label=lab[i], error_kw=dict(lw=1))
    ax.set_yscale("log"); ax.set_xticks(range(7)); ax.set_xticklabels(SHORT); ax.set_ylabel("Validation RMSE (mean ± std)")
    ax.set_title("(a) Matched protocol: 10 hidden units, 150 epochs, same Lyapunov rate", fontsize=10, fontweight="bold"); ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax = axs[1]; arch = ["MRENN", "MRENN-noTheta", "MRENN-noC2", "MRENN-noWd", "MRENN-lag2", "MRENN-fixedLR"]
    lab = ["full MRENN", "no direct path Θ", "no context 2", "no gain $W_d$", "regressor lag 2", "fixed rate"]; cols = [C["MRENN"], "#8a2be2", "#00b8d9", "#ff8a00", "#12a02a", "#555555"]; w = .13
    for i, a in enumerate(arch):
        m = np.array([rm(a, n)["mean"] for n in NAMES]); s = np.array([rm(a, n)["std"] for n in NAMES])
        div = ~np.isfinite(m); m2 = np.where(div, 30, m)
        ax.bar(np.arange(7) + (i - 2.5) * w, m2, w, yerr=np.where(div, 0, s), color=cols[i], edgecolor="k", lw=.5, capsize=2, label=lab[i], error_kw=dict(lw=1))
        for x in np.arange(7)[div]: ax.text(x + (i - 2.5) * w, 40, "div.", ha="center", fontsize=7, rotation=90, color="#555")
    ax.set_yscale("log"); ax.set_ylim(.01, 120); ax.set_xticks(range(7)); ax.set_xticklabels(SHORT); ax.set_ylabel("Validation RMSE (mean ± std)")
    ax.set_title("(b) MRENN component ablation (same budget)", fontsize=10, fontweight="bold"); ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT + "fig17_matched_ablation.png", dpi=300); plt.close(fig)


def fig_noise():
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.6)); axs = axs.ravel(); lv = ["0", "0.05", "0.15", "0.3"]; xs = [0, 5, 15, 30]
    for p, sub in enumerate(SUBS):
        ax = axs[p]; key, ch = sub
        for a in ["FNN", "ENN", "MRENN"]:
            M = []
            for (k, s), d in sorted(Dm.items()):
                if k != key: continue
                pc = d["archs"][a]["per_channel"][ch]
                M.append([pc["sp"]["rmse"]] + [pc["noise"][l]["rmse"] for l in lv[1:]])
            M = np.array(M); m, sd = M.mean(0), M.std(0, ddof=1)
            ax.errorbar(xs, m, sd, color=C[a], ls=LS[a], marker="o", ms=4, lw=1.8, capsize=3, label=a)
        ax.set_yscale("log"); ax.set_title(NAMES[p], fontsize=11, fontweight="bold"); ax.set_xlabel("Noise level (% of signal std)"); ax.set_xticks(xs)
        if p % 4 == 0: ax.set_ylabel("Validation RMSE")
    axs[7].axis("off"); axs[7].legend(handles=[Line2D([], [], color=C[a], ls=LS[a], marker="o", label=a) for a in ["FNN", "ENN", "MRENN"]], loc="center", frameon=False, fontsize=11)
    fig.tight_layout(); fig.savefig(OUT + "fig18_noise_robustness.png", dpi=300); plt.close(fig)


def fig_complexity():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3)); w = .26
    def g(a, sub, f):
        key, ch = sub; return np.mean([d["archs"][a][f] for (k, s), d in Dm.items() if k == key])
    for j, (f, yl, ttl) in enumerate([("n_params", "Trainable parameters", "(a) Parameter count"), ("macs", "Multiply–accumulates per step", "(b) Forward cost"), ("train_time_s", "Training time per run (s)", "(c) Training time (NumPy, 1 core)")]):
        ax = axs[j]
        for i, a in enumerate(["FNN", "ENN", "MRENN"]):
            v = [g(a, s, f) for s in SUBS]; ax.bar(np.arange(7) + (i - 1) * w, v, w, color=C[a], edgecolor="k", lw=.6, label=a)
        ax.set_xticks(range(7)); ax.set_xticklabels(SHORT, rotation=30); ax.set_ylabel(yl); ax.set_title(ttl, fontsize=10, fontweight="bold")
        if j == 2: ax.set_yscale("log")
    axs[0].legend(frameon=False); fig.tight_layout(); fig.savefig(OUT + "fig19_complexity.png", dpi=300); plt.close(fig)


def fig_lyap():
    L = json.load(open(RES + "/lyap_diag.json")); fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    rhos = ["0.1", "1.0", "1.9"]; cols = ["#1f6bff", "#12a02a", "#e3202b"]
    for ax, key in zip(axs, ["ex1", "ex3"]):
        data = [np.array(L[f"{key}|{r}"]["sample"])[:, 1] for r in rhos]
        bp = ax.boxplot(data, positions=range(3), widths=.5, patch_artist=True, showfliers=True, flierprops=dict(marker="o", ms=3, alpha=.6))
        for patch, c in zip(bp["boxes"], cols): patch.set_facecolor(c); patch.set_alpha(.55)
        for med in bp["medians"]: med.set_color("k")
        ax.scatter(range(3), [abs(1 - float(r)) for r in rhos], marker="_", s=900, color="k", zorder=5, label=r"linearised prediction $|1-\rho|$")
        ax.axhline(1, color="#888", ls="--", lw=1); ax.text(2.45, 1.02, "no descent", fontsize=8, ha="right", color="#555")
        ax.set_xticks(range(3)); ax.set_xticklabels([fr"$\rho$ = {r}" for r in rhos]); ax.set_title({"ex1": "(a) Example 1", "ex3": "(b) Example 3"}[key], fontsize=10, fontweight="bold")
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    axs[0].set_ylabel(r"measured ratio $|e^{+}(k)|/|e(k)|$")
    fig.tight_layout(); fig.savefig(OUT + "fig20_lyapunov_check.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    fig_convergence(); fig_val_summary(); fig_examples(); fig_fr(); fig_matched_ablation(); fig_noise(); fig_complexity(); fig_lyap()
