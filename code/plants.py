"""Narendra-Parthasarathy (1990) benchmark plants, Examples 1-6.

Time-index convention used throughout the package (and the manuscript):
the plant is  y_p(k+1) = F[y_p(k), y_p(k-1), ..., u(k), u(k-1), ...]  with
y_p(k) = 0 and u(k) = 0 for k <= 0 (plant starts at rest).  The returned
array y satisfies y[k] = y_p(k), so  y[k+1] = F(y[k], y[k-1], ..., u[k], ...).
"""
import numpy as np

PAD = 3  # enough leading zeros for the deepest regressor (order 3)


def _f_ex1(u):
    return 0.6 * np.sin(np.pi * u) + 0.3 * np.sin(3 * np.pi * u) + 0.1 * np.sin(5 * np.pi * u)


def _f_ex6(u):
    return (u - 0.8) * u * (u + 0.5)


# order of each plant: (n_y, n_u); u_range = training excitation range
PLANTS = {
    "ex1": dict(n_y=2, n_u=1, u_range=(-1.0, 1.0), mimo=False, label="Example 1"),
    "ex2": dict(n_y=2, n_u=1, u_range=(-2.0, 2.0), mimo=False, label="Example 2"),
    "ex3": dict(n_y=1, n_u=1, u_range=(-2.0, 2.0), mimo=False, label="Example 3"),
    "ex4": dict(n_y=3, n_u=2, u_range=(-1.0, 1.0), mimo=False, label="Example 4"),
    "ex5": dict(n_y=1, n_u=1, u_range=(-1.0, 1.0), mimo=True, label="Example 5"),
    "ex6": dict(n_y=1, n_u=1, u_range=(-1.0, 1.0), mimo=False, label="Example 6"),
}


def simulate(key, u):
    """Return y with y[k] = y_p(k), length len(u)+1 (shape (N+1,) or (N+1,2))."""
    N = len(u)
    if key == "ex5":
        y = np.zeros((N + 1, 2))
        for k in range(N):
            y1, y2 = y[k]
            den = 1.0 + y2 * y2
            y[k + 1, 0] = y1 / den + u[k, 0]
            y[k + 1, 1] = y1 * y2 / den + u[k, 1]
        return y
    yp = np.zeros(N + 1 + PAD)  # yp[k+PAD] = y_p(k)
    up = np.concatenate([np.zeros(PAD), np.asarray(u, float)])  # up[k+PAD] = u(k)
    for k in range(N):
        i = k + PAD
        if key == "ex1":
            v = 0.3 * yp[i] + 0.6 * yp[i - 1] + _f_ex1(up[i])
        elif key == "ex2":
            y1, y2 = yp[i], yp[i - 1]
            v = y1 * y2 * (y1 + 2.5) / (1 + y1 * y1 + y2 * y2) + up[i]
        elif key == "ex3":
            v = yp[i] / (1 + yp[i] ** 2) + up[i] ** 3
        elif key == "ex4":
            y1, y2, y3 = yp[i], yp[i - 1], yp[i - 2]
            v = (y1 * y2 * y3 * up[i - 1] * (y3 - 1) + up[i]) / (1 + y2 * y2 + y3 * y3)
        elif key == "ex6":
            v = 0.8 * yp[i] + _f_ex6(up[i])
        else:
            raise KeyError(key)
        yp[i + 1] = v
    return yp[PAD:]


def make_data(key, seed, n_train=None, n_val=None):
    """Training excitation (uniform white), validation excitation (multi-segment
    sinusoid, Kumar 2023 eq. 63 style) and the corresponding plant outputs."""
    spec = PLANTS[key]
    rng = np.random.default_rng(seed)
    lo, hi = spec["u_range"]
    n_train = n_train or (800 if spec["mimo"] else 1500)
    n_val = n_val or (800 if spec["mimo"] else 1200)
    t = np.arange(n_val)
    if spec["mimo"]:
        u_tr = rng.uniform(lo, hi, (n_train, 2))
        u_va = np.stack([np.sin(2 * np.pi * t / 25), np.cos(2 * np.pi * t / 25)], axis=1)
    else:
        u_tr = rng.uniform(lo, hi, n_train)
        u_va = np.sin(2 * np.pi * t / 250)
        u_va[500:] = 0.8 * np.sin(2 * np.pi * t[500:] / 250) + 0.2 * np.sin(2 * np.pi * t[500:] / 25)
    return u_tr, simulate(key, u_tr), u_va, simulate(key, u_va)
