import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Polygon
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 11})
import os; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures") + os.sep
BLUE, GREEN, RED, PURPLE, ORANGE, GREY = "#1f6bff", "#12a02a", "#e3202b", "#8a2be2", "#ff8a00", "#9aa0aa"


def node(ax, xy, r, label="", fc="white", ec="black", lw=1.6, ls="-", fs=9):
    ax.add_patch(Circle(xy, r, fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
    if label: ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=fs, zorder=4)


def line(ax, a, b, c=GREY, lw=0.8, ls="-", z=1, alpha=1):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=lw, ls=ls, zorder=z, alpha=alpha)


def arrow(ax, a, b, c="black", lw=1.6, ls="-", style="-|>", rad=0.0, z=2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14, color=c, lw=lw, ls=ls, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def selfloop(ax, xy, r, c, label, side="left", fs=9):
    x, y = xy; dx = -1 if side == "left" else 1
    th = np.linspace(0, 2 * np.pi, 80); R = 1.5
    ax.plot(x + dx * (r + R - 0.2) + R * np.cos(th), y + R * np.sin(th), color=c, lw=1.4, zorder=2)
    ax.text(x + dx * (r + R - 0.2), y + R + 0.6, label, color=c, fontsize=fs, ha="center", va="bottom")


def setup(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, w * 10); ax.set_ylim(0, h * 10); ax.axis("off"); ax.set_aspect("equal"); return fig, ax


def fnn():
    fig, ax = setup(8.4, 4.4)
    ins = ["$y_p(k)$", "$y_p(k{-}1)$", "$y_p(k{-}2)$", "$u(k)$", "$u(k{-}1)$"]
    yi = np.linspace(38, 8, 5); yh = np.linspace(40, 6, 6)
    for i, lab in enumerate(ins):
        node(ax, (14, yi[i]), 1.6); ax.text(10.5, yi[i], lab, ha="right", va="center", fontsize=11)
    for j in range(6):
        for i in range(5): line(ax, (15.6, yi[i]), (42.4, yh[j]), alpha=.55)
        node(ax, (44, yh[j]), 1.9, "tanh", fc="#e8f0ff", ec=BLUE, fs=7.5)
        line(ax, (45.9, yh[j]), (68.3, 23), c=BLUE, lw=1.3)
    node(ax, (70, 23), 2.0, r"$\Sigma$", fs=12); arrow(ax, (72, 23), (80, 23)); ax.text(81, 23, r"$\hat{y}(k{+}1)$", va="center", fontsize=12)
    ax.text(30, 46, r"$W^{1}_{ij}$", color="#555", fontsize=11, ha="center"); ax.text(58, 38, r"$W^{2}_{j}$", color=BLUE, fontsize=11)
    ax.text(14, 1.5, "Tapped-delay-line regressor $x(k)$", ha="center", fontsize=9.5)
    ax.text(44, 1.5, "Hidden layer, tanh (+$b^1_j$), Eq. (12)", ha="center", fontsize=9.5)
    ax.text(73, 1.5, "Linear output (+$b^2$), Eq. (13)", ha="center", fontsize=9.5)
    fig.savefig(OUT + "fig02_fnn_arch.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def enn():
    fig, ax = setup(8.4, 5.4)
    ins = ["$y_p(k)$", "$y_p(k{-}1)$", "$y_p(k{-}2)$", "$u(k)$", "$u(k{-}1)$"]
    yi = np.linspace(48, 18, 5); yh = np.linspace(50, 16, 6)
    for i, lab in enumerate(ins):
        node(ax, (14, yi[i]), 1.6); ax.text(10.5, yi[i], lab, ha="right", va="center", fontsize=11)
    for j in range(6):
        for i in range(5): line(ax, (15.6, yi[i]), (42.4, yh[j]), alpha=.55)
        node(ax, (44, yh[j]), 1.9, "tanh", fc="#e6f7e8", ec=GREEN, fs=7.5)
        line(ax, (45.9, yh[j]), (68.3, 33), c=GREEN, lw=1.3)
    xc = [36, 44, 52]
    for x in xc:
        node(ax, (x, 5), 1.9, r"$z^{-1}$", fc="#f2f2f2", ec=GREEN, ls="--", fs=8)
        for j in range(6): line(ax, (x, 6.9), (44, yh[j] - 1.9), c=GREEN, lw=0.7, ls="--", alpha=.6)
    node(ax, (70, 33), 2.0, r"$\Sigma$", fs=12); arrow(ax, (72, 33), (80, 33)); ax.text(81, 33, r"$\hat{y}(k{+}1)$", va="center", fontsize=12)
    ax.text(30, 56, r"$W^{i}_{ij}$", color="#555", fontsize=11, ha="center"); ax.text(58, 47, r"$W^{o}_{j}$", color=GREEN, fontsize=11)
    ax.text(25, 5, "Context layer $c(k)=s(k{-}1)$", color=GREEN, ha="center", fontsize=9.5); ax.text(44, 11.3, r"$W^{c}_{lj}$", color=GREEN, fontsize=10, ha="center")
    ax.text(14, 1.2, "Order-correct regressor $x(k)$", ha="center", fontsize=9.5)
    ax.text(62, 1.2, "Hidden layer Eqs. (14), (15)  |  linear output Eq. (16)", ha="center", fontsize=9.5)
    fig.savefig(OUT + "fig03_enn_arch.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def mrenn():
    fig, ax = setup(10.0, 6.4)
    yi = [40, 22]; labs = ["$y_p(k)$", "$u(k)$"]; yh = np.linspace(52, 20, 5)
    for i in range(2):
        node(ax, (10, yi[i]), 1.7); ax.text(6.5, yi[i], labs[i], ha="right", va="center", fontsize=12)
    for j in range(5):
        for i in range(2): line(ax, (11.7, yi[i]), (42.4, yh[j]), alpha=.6)
        node(ax, (44, yh[j]), 1.9, "tanh", fc="#fde8ea", ec=RED, fs=7.5)
        line(ax, (45.9, yh[j]), (78.3, 36), c=RED, lw=1.3)
    # direct path
    ax.plot([11.7, 18, 18, 78, 78], [40, 40, 61, 61, 38.2], color=PURPLE, lw=1.8, zorder=2); arrow(ax, (78, 40), (78, 38.2), c=PURPLE)
    ax.text(48, 62.2, r"direct input–output path $\Theta$, Eq. (19)", color=PURPLE, ha="center", fontsize=10.5)
    # context 1
    xc = [34, 44, 54]
    for q, x in enumerate(xc):
        node(ax, (x, 6), 1.9, f"$c1_{q+1}$", fc="#f0eaff", ec=RED, ls="--", fs=8)
        selfloop(ax, (x, 6), 1.9, RED, f"$W_{{d{q+1}}}$", side="left")
        for j in range(5): line(ax, (x, 7.9), (44, yh[j] - 1.9), c=RED, lw=0.6, ls="--", alpha=.55)
    ax.text(44, 14.2, r"$W_{cj}$", color=RED, fontsize=10, ha="center")
    ax.text(40, -3.8, "context layer 1, Eq. (18):\n$c1_q(k)=s_q(k{-}1)+W_{dq}\,c1_q(k{-}1)$", ha="center", va="top", fontsize=9.5, color="#333")
    node(ax, (80, 36), 2.1, r"$\Sigma$", fs=12); arrow(ax, (82.1, 36), (92, 36)); ax.text(93, 36, r"$\hat{y}(k{+}1)$", va="center", fontsize=12)
    ax.text(62, 47, r"$W_{oi}$", color=RED, fontsize=11)
    # context 2
    node(ax, (80, 12), 1.9, "$c2_1$", fc="#fde8ea", ec=RED, ls="--", fs=8); selfloop(ax, (80, 12), 1.9, RED, r"$W_f$", side="left")
    arrow(ax, (80, 14), (80, 33.9), c=RED); ax.text(82, 24, r"$W_{oc}$", color=RED, fontsize=11)
    ax.plot([87, 87], [36, 12], color=RED, lw=1.4, ls="--"); arrow(ax, (87, 12), (82, 12), c=RED, ls="--")
    ax.text(88.5, 24, r"$\hat{y}(k)$", color=RED, fontsize=11)
    ax.text(83, -3.8, "context layer 2, Eq. (20):\n$c21(k)=\\hat{y}(k)+W_f\,c21(k{-}1)$", ha="center", va="top", fontsize=9.5, color="#333")
    ax.text(10, 8, "two external inputs\n(four in the MIMO case)", fontsize=9, color="#444", ha="center", va="center")
    ax.text(76, 57, "biases enter with a plus sign:\n$+b_j$ (hidden), $+b_o$ (output)", fontsize=9, bbox=dict(fc="#f6f4ee", ec="#cfcab8", boxstyle="round,pad=0.4"), ha="center", va="center")
    ax.set_ylim(-14, 66); ax.set_xlim(0, 100)
    fig.savefig(OUT + "fig04_mrenn_arch.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def box(ax, xy, w, h, txt, fc="white", ec="black", fs=10, sub=None):
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.6, zorder=3))
    ax.text(xy[0], xy[1] + (1.2 if sub else 0), txt, ha="center", va="center", fontsize=fs, zorder=4)
    if sub: ax.text(xy[0], xy[1] - 2.4, sub, ha="center", va="center", fontsize=fs - 2, color="#555", zorder=4)


def seriespar():
    fig, ax = setup(10.8, 4.2); ax.set_xlim(0, 108); ax.set_ylim(-4, 42)
    box(ax, (30, 33), 22, 8, "Nonlinear plant", "#eef1f6", sub="Eqs. (2)–(11)")
    box(ax, (30, 14), 22, 8, "Regressor $x(k)$", sub="Table 2")
    box(ax, (64, 14), 28, 8, "Neural identifier", "#fbfbf3", sub="FNN / ENN / MRENN")
    node(ax, (96, 24), 2.6, r"$\Sigma$", fs=12)
    arrow(ax, (4, 33), (19, 33)); ax.text(2, 36, "$u(k)$", fontsize=12); ax.plot([9, 9], [33, 14], color="black", lw=1.6); arrow(ax, (9, 14), (19, 14))
    arrow(ax, (41, 33), (96, 33)); arrow(ax, (96, 33), (96, 26.7)); ax.text(70, 35.2, r"$y_p(k{+}1)$", fontsize=12); ax.text(93.4, 29.4, "+", fontsize=12)
    ax.plot([52, 52], [33, 25], color="#6f7468", lw=1.5, ls="--"); arrow(ax, (52, 25), (30, 25), c="#6f7468", ls="--"); arrow(ax, (30, 25), (30, 18), c="#6f7468", ls="--")
    ax.text(53.5, 27.5, "series-parallel: true output history", fontsize=9, color="#555")
    arrow(ax, (41, 14), (50, 14)); arrow(ax, (78, 14), (96, 14)); arrow(ax, (96, 14), (96, 21.4)); ax.text(84, 15.8, r"$\hat{y}(k{+}1)$", fontsize=12); ax.text(93.4, 17.4, r"$-$", fontsize=12)
    arrow(ax, (99, 24), (106, 24)); ax.text(100, 26.2, r"$e(k)$", fontsize=12)
    # parallel (free-running) path
    ax.plot([86, 86, 30], [14, 3, 3], color=ORANGE, lw=1.8, ls="--"); arrow(ax, (30, 3), (30, 9.8), c=ORANGE, ls="--")
    ax.text(58, 0.2, "parallel (free-running) evaluation: own predictions $\\hat{y}$ replace the true history", fontsize=9, color=ORANGE, ha="center")
    ax.text(64, -3.2, "parameter update (training only): Adam / delta rule / BPTT / Lyapunov-normalized $\\eta(k)$, Section VII", fontsize=9, color=RED, ha="center")
    fig.savefig(OUT + "fig05_series_parallel.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def flow():
    fig, ax = setup(7.0, 9.4); ax.set_xlim(-2, 70); ax.set_ylim(0, 94)
    steps = [("Start", "#eef1f6", None), ("Draw excitation $u(k)$, seed $s$", "white", "input range in Table 1; 1500 samples (800 MIMO)"),
             ("Simulate plant $y_p(k{+}1)$", "white", "Eqs. (2)–(11); plant starts at rest"),
             ("Build regressor $x(k)$", "white", "true $y_p$ history (series-parallel), Table 2"),
             ("Initialise weights and states", "#e8f0ff", "Sec. VIII-A; $c1=c2=0$ at the start of every epoch"),
             ("Forward pass and error $e(k)$", "#e8f0ff", "Eqs. (12)–(20);  $e=y_p(k{+}1)-\\hat{y}(k{+}1)$"),
             ("Parameter update", "#fde8ea", "Adam | delta rule | BPTT | Lyapunov $\\eta(k)$, Eq. (27)"),
             ("Project memory gains", "#fde8ea", "$|W_d|\\leq 0.95$,  $|W_f+W_{oc}|\\leq 0.95$"),
             ("Epoch budget reached?", "#fff4dd", "Table 4"),
             ("Validate: series-parallel and free-running", "white", "held-out sinusoid, 1200 samples (800 MIMO)"),
             ("RMSE and MAE; repeat for 5 seeds", "white", "mean, std, 95 % CI, Wilcoxon, Friedman"),
             ("Noise, out-of-range and ablation tests", "#f0f0e8", "post-training, no retraining")]
    ys = [88, 79.5, 71, 62.5, 54, 45.5, 37, 28.5, 18.5, 8.5, -1.5, -11.5]
    ys = [y + 7 for y in ys[:len(steps)]]
    for (t, fc, sub), y in zip(steps, ys):
        if t.startswith("Epoch"):
            ax.add_patch(Polygon([[35, y + 5], [58, y], [35, y - 5], [12, y]], closed=True, fc=fc, ec="black", lw=1.6, zorder=3))
            ax.text(35, y + 0.9, t, ha="center", va="center", fontsize=9, zorder=4); ax.text(35, y - 1.8, sub, ha="center", va="center", fontsize=7.5, color="#555", zorder=4)
        else:
            ax.add_patch(FancyBboxPatch((13, y - 3.0), 44, 6.0, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec="black", lw=1.6, zorder=3))
            ax.text(35, y + (0.9 if sub else 0), t, ha="center", va="center", fontsize=9, zorder=4)
            if sub: ax.text(35, y - 1.5, sub, ha="center", va="center", fontsize=7.5, color="#555", zorder=4)
    ext = lambda i, up: (5.0 if steps[i][0].startswith("Epoch") else 3.3)
    for i in range(len(ys) - 1): arrow(ax, (35, ys[i] - ext(i, 0)), (35, ys[i + 1] + ext(i + 1, 1)), lw=1.3)
    yE, yU = ys[8], ys[6]
    ax.plot([12, 4, 4], [yE, yE, yU], color=RED, lw=1.4); arrow(ax, (4, yU), (12.6, yU), c=RED, lw=1.4)
    ax.text(2.2, (yE + yU) / 2, "no", color=RED, rotation=90, va="center", fontsize=9)
    ax.text(37.0, yE - 5.7, "yes", fontsize=8, va="center", ha="left", bbox=dict(fc="white", ec="none", pad=0.6), zorder=6)
    ax.set_ylim(ys[-1] - 5, 99)
    fig.savefig(OUT + "fig06_train_val_procedure.png", dpi=300, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fnn(); enn(); mrenn(); seriespar(); flow()
