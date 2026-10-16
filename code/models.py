"""FNN, Elman (delta rule / Lyapunov-rate / truncated-BPTT) and MRENN identifiers.
NumPy only; all gradients are derived by hand.  Every model maps a regressor x(k)
to yhat(k+1) (estimate of y_p(k+1)).
"""
import numpy as np

tanh = np.tanh


# ----------------------------------------------------------------------------- FNN
class FNN:
    kind = "full"

    def __init__(self, n_in, H, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, .5, (n_in, H)); self.b1 = np.zeros(H)
        self.W2 = r.normal(0, .5, H); self.b2 = np.zeros(1)
        self.H, self.n_in = H, n_in
        self.n_params = n_in * H + H + H + 1
        self.state_floats = 0

    def reset(self): pass

    def predict(self, x):                      # x: (n_in,) or (n,n_in)
        return tanh(x @ self.W1 + self.b1) @ self.W2 + self.b2[0]

    step = predict

    def macs(self):                            # multiply-accumulates per forward step
        return self.n_in * self.H + self.H

    # full-batch Adam (as in Narendra & Parthasarathy's static back-propagation setting)
    def fit_adam(self, X, Y, epochs=800, lr=0.02, b1=.9, b2=.999, eps=1e-8):
        P = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]
        n = len(X); curve = []
        for t in range(1, epochs + 1):
            S = tanh(X @ P[0] + P[1]); e = S @ P[2] + P[3][0] - Y
            curve.append(float(np.sqrt(np.mean(e ** 2))))
            d = np.outer(e, P[2]) * (1 - S ** 2)
            G = [X.T @ d / n, d.mean(0), S.T @ e / n, np.array([e.mean()])]
            for i in range(4):
                m[i] = b1 * m[i] + (1 - b1) * G[i]; v[i] = b2 * v[i] + (1 - b2) * G[i] ** 2
                P[i] -= lr * (m[i] / (1 - b1 ** t)) / (np.sqrt(v[i] / (1 - b2 ** t)) + eps)
        return curve

    # sequential training with the same Lyapunov-normalised rate used for MRENN
    def fit_lyap(self, X, Y, epochs=150, rho=.25, eps=1e-3):
        curve = []
        for _ in range(epochs):
            se = 0.0
            for x, y in zip(X, Y):
                s = tanh(x @ self.W1 + self.b1); yh = s @ self.W2 + self.b2[0]; e = y - yh
                dS = 1 - s * s; gb = self.W2 * dS
                gW1 = x[:, None] * gb[None, :]
                en = s @ s + 1.0 + (gW1 * gW1).sum() + gb @ gb
                eta = rho / (eps + en) * e
                self.W2 += eta * s; self.b2 += eta; self.W1 += eta * gW1; self.b1 += eta * gb
                se += e * e
            curve.append(float(np.sqrt(se / len(X))))
        return curve


# --------------------------------------------------------------------------- Elman
class Elman:
    kind = "full"

    def __init__(self, n_in, H, seed=0):
        r = np.random.default_rng(seed)
        self.Wi = r.normal(0, .5, (n_in, H)); self.Wc = r.normal(0, .5, (H, H))
        self.b = np.zeros(H); self.Wo = r.normal(0, .5, H); self.bo = np.zeros(1)
        self.H, self.n_in = H, n_in
        self.n_params = n_in * H + H * H + H + H + 1
        self.state_floats = H
        self.c = np.zeros(H)

    def reset(self): self.c = np.zeros(self.H)

    def step(self, x):
        s = tanh(x @ self.Wi + self.c @ self.Wc + self.b)
        self.c = s
        return s @ self.Wo + self.bo[0]

    def macs(self): return self.n_in * self.H + self.H * self.H + self.H

    # classical delta rule, context treated as a constant input (no BPTT), momentum
    def fit_delta(self, X, Y, epochs=300, lr=.035, mom=.9, clip=5.0):
        V = {k: np.zeros_like(getattr(self, k)) for k in ("Wi", "Wc", "b", "Wo", "bo")}
        curve = []
        for _ in range(epochs):
            self.reset(); se = 0.0
            for x, y in zip(X, Y):
                s = tanh(x @ self.Wi + self.c @ self.Wc + self.b); yh = s @ self.Wo + self.bo[0]
                e = yh - y; d = e * self.Wo * (1 - s * s)
                G = {"Wo": np.clip(e * s, -clip, clip), "bo": np.clip(np.array([e]), -clip, clip),
                     "Wi": np.clip(x[:, None] * d[None, :], -clip, clip),
                     "b": np.clip(d, -clip, clip), "Wc": np.clip(self.c[:, None] * d[None, :], -clip, clip)}
                for k, g in G.items():
                    V[k] = mom * V[k] - lr * g
                    setattr(self, k, getattr(self, k) + V[k])
                self.c = s; se += e * e
            curve.append(float(np.sqrt(se / len(X))))
        return curve

    # same Lyapunov-normalised rate as MRENN, context frozen (isolates the architecture)
    def fit_lyap(self, X, Y, epochs=150, rho=.25, eps=1e-3):
        curve = []
        for _ in range(epochs):
            self.reset(); se = 0.0
            for x, y in zip(X, Y):
                s = tanh(x @ self.Wi + self.c @ self.Wc + self.b); yh = s @ self.Wo + self.bo[0]; e = y - yh
                gb = self.Wo * (1 - s * s)
                gWi = x[:, None] * gb[None, :]; gWc = self.c[:, None] * gb[None, :]
                en = s @ s + 1.0 + (gWi * gWi).sum() + (gWc * gWc).sum() + gb @ gb
                eta = rho / (eps + en) * e
                self.Wo += eta * s; self.bo += eta; self.Wi += eta * gWi; self.Wc += eta * gWc; self.b += eta * gb
                self.c = s; se += e * e
            curve.append(float(np.sqrt(se / len(X))))
        return curve

    # truncated back-propagation through time (window T), Adam, gradient-norm clipping
    def fit_bptt(self, X, Y, epochs=300, T=20, lr=.005, b1=.9, b2=.999, eps=1e-8, gclip=5.0):
        names = ("Wi", "Wc", "b", "Wo", "bo")
        m = {k: np.zeros_like(getattr(self, k)) for k in names}; v = {k: np.zeros_like(getattr(self, k)) for k in names}
        t_ad = 0; curve = []; N = len(X)
        for _ in range(epochs):
            self.reset(); se = 0.0
            for a in range(0, N, T):
                xs, ys = X[a:a + T], Y[a:a + T]; n = len(xs)
                S = np.zeros((n, self.H)); C = np.zeros((n, self.H)); E = np.zeros(n)
                c = self.c
                for t in range(n):
                    C[t] = c
                    s = tanh(xs[t] @ self.Wi + c @ self.Wc + self.b)
                    S[t] = s; E[t] = s @ self.Wo + self.bo[0] - ys[t]; c = s
                se += float((E ** 2).sum())
                g = {k: np.zeros_like(getattr(self, k)) for k in names}
                carry = np.zeros(self.H)
                for t in range(n - 1, -1, -1):
                    dy = E[t] / n
                    g["Wo"] += dy * S[t]; g["bo"] += dy
                    dpre = (dy * self.Wo + carry) * (1 - S[t] ** 2)
                    g["Wi"] += xs[t][:, None] * dpre[None, :]; g["Wc"] += C[t][:, None] * dpre[None, :]; g["b"] += dpre
                    carry = self.Wc @ dpre
                nrm = np.sqrt(sum((q ** 2).sum() for q in g.values()))
                if nrm > gclip:
                    for k in g: g[k] *= gclip / nrm
                t_ad += 1
                for k in names:
                    m[k] = b1 * m[k] + (1 - b1) * g[k]; v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                    setattr(self, k, getattr(self, k) - lr * (m[k] / (1 - b1 ** t_ad)) / (np.sqrt(v[k] / (1 - b2 ** t_ad)) + eps))
                self.c = c
            curve.append(float(np.sqrt(se / N)))
        return curve


# --------------------------------------------------------------------------- MRENN
class MRENN:
    """Memory recurrent Elman network, diagonal context layer, eqs. (11)-(15) of the manuscript.

    c1_q(k)  = s_q(k-1) + Wd_q c1_q(k-1)                      (context layer 1)
    c21(k)   = yhat(k)  + Wf   c21(k-1)                       (context layer 2)
    s_j(k)   = tanh( sum_u x_u(k) Wi_uj + c1_j(k) Wc_j + b_j )
    yhat(k+1)= sum_j s_j(k) Wo_j + sum_u x_u(k) Theta_u + c21(k) Woc + bo
    """
    kind = "min"

    def __init__(self, n_in, H, seed=0, use_theta=True, use_c2=True, use_wd=True,
                 wmax=.95):
        r = np.random.default_rng(seed)
        self.Wi = r.normal(0, .4, (n_in, H)); self.Wc = r.normal(0, .4, H); self.b = r.normal(0, .1, H)
        self.Wd = r.uniform(-.3, .3, H); self.Wo = r.normal(0, .4, H); self.Theta = r.normal(0, .2, n_in)
        self.Woc = r.normal(0, .4); self.bo = 0.0; self.Wf = r.uniform(-.3, .3)
        if not use_theta: self.Theta[:] = 0
        if not use_c2: self.Woc = 0.0; self.Wf = 0.0
        if not use_wd: self.Wd[:] = 0
        self.flags = (use_theta, use_c2, use_wd)
        self.H, self.n_in, self.wmax = H, n_in, wmax
        self.n_params = n_in * H + H + H + H + H + n_in + 1 + 1 + 1   # Wi,Wc,b,Wd,Wo,Theta,Woc,bo,Wf
        self.state_floats = 2 * H + 2
        self.stats = dict(steps=0, descent=0, ratio_max=0.0, ratio_sum=0.0, proj_wd=0, proj_loop=0)
        self.log = None
        self.reset()
        self.project()

    def reset(self):
        self.s_prev = np.zeros(self.H); self.c1_prev = np.zeros(self.H)
        self.y_prev = 0.0; self.c2_prev = 0.0

    def macs(self):
        return self.n_in * self.H + self.H + self.H + self.n_in + 1 + self.H + 1

    def project(self):
        """Keep both memory loops contractive: |Wd_q| <= wmax and |Wf + Woc| <= wmax."""
        use_theta, use_c2, use_wd = self.flags
        if use_wd:
            n = int((np.abs(self.Wd) > self.wmax).sum()); self.Wd = np.clip(self.Wd, -self.wmax, self.wmax)
            self.stats["proj_wd"] += n
        if use_c2:
            self.Wf = float(np.clip(self.Wf, -self.wmax, self.wmax))
            lo, hi = -self.wmax - self.Wf, self.wmax - self.Wf
            if not (lo <= self.Woc <= hi):
                self.stats["proj_loop"] += 1
            self.Woc = float(np.clip(self.Woc, lo, hi))

    def _forward(self, x, Wi, Wc, b, Wd, Wo, Th, Woc, bo, Wf, c1p, c2p, sp, yp):
        c1 = sp + Wd * c1p; c2 = yp + Wf * c2p
        s = tanh(x @ Wi + c1 * Wc + b)
        return s @ Wo + x @ Th + c2 * Woc + bo, s, c1, c2

    def step(self, x, y_target=None, rho=.25, eps=1e-3, eta_fixed=None, diag=False):
        yh, s, c1, c2 = self._forward(x, self.Wi, self.Wc, self.b, self.Wd, self.Wo, self.Theta, self.Woc, self.bo,
                                      self.Wf, self.c1_prev, self.c2_prev, self.s_prev, self.y_prev)
        if y_target is not None:
            use_theta, use_c2, use_wd = self.flags
            e = y_target - yh; dS = 1 - s * s; wd_ = self.Wo * dS
            gWi = x[:, None] * wd_[None, :]; gWc = c1 * wd_; gb = wd_
            gWd = wd_ * self.Wc * self.c1_prev if use_wd else None
            en = s @ s + 1.0 + (gWi * gWi).sum() + gWc @ gWc + gb @ gb
            if use_theta: en += x @ x
            if use_c2:
                gWf = self.Woc * self.c2_prev; en += c2 * c2 + gWf * gWf
            if use_wd: en += gWd @ gWd
            eta = (rho / (eps + en)) if eta_fixed is None else eta_fixed
            k = eta * e
            if diag:
                old = (self.Wi.copy(), self.Wc.copy(), self.b.copy(), self.Wd.copy(), self.Wo.copy(), self.Theta.copy(),
                       self.Woc, self.bo, self.Wf)
            self.Wo = self.Wo + k * s; self.bo = self.bo + k
            self.Wi = self.Wi + k * gWi; self.Wc = self.Wc + k * gWc; self.b = self.b + k * gb
            if use_theta: self.Theta = self.Theta + k * x
            if use_c2:
                self.Woc = self.Woc + k * c2; self.Wf = self.Wf + k * gWf
            if use_wd: self.Wd = self.Wd + k * gWd
            if diag and e != 0.0:                       # a-posteriori error, same step, before projection
                yp2 = self._forward(x, self.Wi, self.Wc, self.b, self.Wd, self.Wo, self.Theta, self.Woc, self.bo,
                                    self.Wf, self.c1_prev, self.c2_prev, self.s_prev, self.y_prev)[0]
                r = abs(y_target - yp2) / abs(e)
                st = self.stats; st["steps"] += 1; st["descent"] += int(r < 1.0)
                st["ratio_max"] = max(st["ratio_max"], r); st["ratio_sum"] += r
                if self.log is not None: self.log.append((eta * en, r, abs(e)))
            self.project()
        self.s_prev, self.c1_prev, self.y_prev, self.c2_prev = s, c1, yh, c2
        return yh

    def fit(self, X, Y, epochs=150, rho=.25, eps=1e-3, eta_fixed=None, diag=False):
        curve = []
        for _ in range(epochs):
            self.reset(); se = 0.0
            for x, y in zip(X, Y):
                yh = self.step(x, y, rho, eps, eta_fixed, diag); se += (y - yh) ** 2
            curve.append(float(np.sqrt(se / len(X))))
        return curve
