"""Shared experiment utilities: model factory, training, series-parallel and free-running evaluation."""
import time
import numpy as np
from plants import PLANTS, make_data
from regressors import build_X, row, n_inputs
from models import FNN, Elman, MRENN

# architecture -> (class, regressor kind, hidden units, training call)
DEFAULT = {
    "FNN":        dict(cls="FNN",    kind="full", H=24, epochs=800, lr=0.02),
    "ENN":        dict(cls="Elman",  kind="full", H=12, epochs=300, lr=0.035, mom=0.9),
    "MRENN":      dict(cls="MRENN",  kind="min",  H=10, epochs=150, rho=0.25, eps=1e-3),
    "ENN-BPTT":   dict(cls="Elman",  kind="full", H=12, epochs=300, lr=0.005, T=20),
    "FNN-m":      dict(cls="FNN",    kind="full", H=10, epochs=150, rho=0.25, eps=1e-3),
    "ENN-m":      dict(cls="Elman",  kind="full", H=10, epochs=150, rho=0.25, eps=1e-3),
}


def make_model(name, n_in, cfg, seed):
    c = cfg["cls"]
    if c == "FNN": return FNN(n_in, cfg["H"], seed)
    if c == "Elman": return Elman(n_in, cfg["H"], seed)
    kw = cfg.get("flags", {})
    return MRENN(n_in, cfg["H"], seed, **kw)


def train(name, cfg, X, Y, seed, diag=False):
    m = make_model(name, X.shape[1], cfg, seed)
    t0 = time.perf_counter()
    if cfg["cls"] == "FNN":
        curve = m.fit_adam(X, Y, cfg["epochs"], cfg["lr"]) if "rho" not in cfg else m.fit_lyap(X, Y, cfg["epochs"], cfg["rho"], cfg["eps"])
    elif cfg["cls"] == "Elman":
        if "T" in cfg: curve = m.fit_bptt(X, Y, cfg["epochs"], cfg["T"], cfg["lr"])
        elif "rho" in cfg: curve = m.fit_lyap(X, Y, cfg["epochs"], cfg["rho"], cfg["eps"])
        else: curve = m.fit_delta(X, Y, cfg["epochs"], cfg["lr"], cfg["mom"])
    else:
        curve = m.fit(X, Y, cfg["epochs"], cfg["rho"], cfg["eps"], cfg.get("eta_fixed"), diag)
    return m, curve, time.perf_counter() - t0


def metrics(pred, true, floor):
    e = pred - true
    ok = np.isfinite(e).all()
    rm = float(np.sqrt(np.mean(e ** 2))) if ok else float("inf")
    mae = float(np.mean(np.abs(e))) if ok else float("inf")
    mape = float(100 * np.mean(np.abs(e) / np.maximum(np.abs(true), floor))) if ok else float("inf")
    return dict(rmse=rm, mae=mae, nrmse=rm / float(true.std()), mape_f=mape)


def sp_predict(m, X):
    m.reset()
    if isinstance(m, FNN): return m.predict(X)
    return np.array([m.step(x) for x in X])


def free_run(models, key, kind, u, y_init=None, cap=1e3):
    """Parallel (free-running) evaluation: every regressor entry that would be a plant
    output is replaced by the model's own earlier prediction.  `models` = list (one per output channel)."""
    mimo = PLANTS[key]["mimo"]
    N = len(u)
    yh = np.zeros((N + 1, 2)) if mimo else np.zeros(N + 1)      # yh[k] = estimate of y_p(k); y_p(0)=0 known
    for m in models: m.reset()
    diverged = False
    for k in range(N):
        x = row(key, kind, yh, u, k)
        for c, m in enumerate(models):
            v = m.step(x)
            if not np.isfinite(v) or abs(v) > cap:
                diverged = True; v = float(np.clip(np.nan_to_num(v, nan=cap), -cap, cap))
            if mimo: yh[k + 1, c] = v
            else: yh[k + 1] = v
    return yh[1:], diverged


def prepare(key, seed):
    u_tr, y_tr, u_va, y_va = make_data(key, seed)
    return u_tr, y_tr, u_va, y_va
