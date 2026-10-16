"""Regressor construction.  At step k every model outputs yhat(k+1).
FNN / Elman ("full"): y_p(k) ... y_p(k-n_y+1), u(k) ... u(k-n_u+1)  (order-correct)
MRENN ("min")       : y_p(k), u(k)   (output delayed by one sample w.r.t. the prediction instant k+1, current input)
"min2" (sensitivity): y_p(k-1), u(k) (output delayed by two samples; used only in the lag ablation)
MIMO (Example 5)    : full = min = [y1(k), y2(k), u1(k), u2(k)];  min2 = [y1(k-1), y2(k-1), u1(k), u2(k)]
Negative time indices are zero (plant starts at rest).
"""
import numpy as np
from plants import PLANTS


def n_inputs(key, kind):
    s = PLANTS[key]
    if s["mimo"]:
        return 4
    return 2 if kind in ("min", "min2") else s["n_y"] + s["n_u"]


def _g(a, i):
    return a[i] if i >= 0 else (0.0 if a.ndim == 1 else np.zeros(a.shape[1]))


def build_X(key, kind, y, u, K):
    """Regressor matrix for steps k = 0..K-1 using histories y (y[k]=y_p(k)) and u."""
    s = PLANTS[key]
    X = np.zeros((K, n_inputs(key, kind)))
    for k in range(K):
        X[k] = row(key, kind, y, u, k)
    return X


def row(key, kind, y, u, k):
    s = PLANTS[key]
    if s["mimo"]:
        ylag = k - 1 if kind == "min2" else k
        return np.concatenate([_g(y, ylag), u[k]])
    if kind == "min":
        return np.array([_g(y, k), u[k]])
    if kind == "min2":
        return np.array([_g(y, k - 1), u[k]])
    yy = [_g(y, k - i) for i in range(s["n_y"])]
    uu = [_g(u, k - i) for i in range(s["n_u"])]
    return np.array(yy + uu)
