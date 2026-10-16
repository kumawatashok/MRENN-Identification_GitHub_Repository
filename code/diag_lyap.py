"""Numerical check of the local-descent theorem: |e+|/|e| versus |1 - alpha|, alpha = eta*||g||^2."""
import numpy as np, json, os
from core import *
np.seterr(all="ignore")
out = {}
for key in ("ex1", "ex3"):
    u, y, uv, yv = prepare(key, 0); K = len(u)
    X = build_X(key, "min", y, u, K); Y = y[1:K + 1]
    for rho in (0.1, 1.0, 1.9):
        m = MRENN(2, 10, 0); m.log = []
        for ep in range(3): m.fit(X, Y, 1, rho=rho, eps=1e-3, diag=True)
        L = np.array(m.log)
        a, r, e = L[:, 0], L[:, 1], L[:, 2]
        dev = np.abs(r - np.abs(1 - a))
        out[f"{key}|{rho}"] = dict(n=int(len(L)), frac_descent=float(np.mean(r < 1)), ratio_max=float(r.max()), dev_median=float(np.median(dev)), dev_max=float(dev.max()),
                                   a_min=float(a.min()), a_max=float(a.max()), sample=L[::97, :2].tolist())
        print(key, rho, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[f'{key}|{rho}'].items() if k != "sample"})
json.dump(out, open(os.path.join("..", "results", "lyap_diag.json"), "w"))
