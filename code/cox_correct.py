"""C1/C2: Cox PH with correct Breslow ties (risk set = all subjects with time >= t, shared by tied events)."""
import numpy as np
from scipy.optimize import minimize


def neg_loglik_grad(beta, x, time, event, alpha=0.0):
    eta = np.clip(x @ beta, -30, 30); w = np.exp(eta)
    ev_times = np.unique(time[event == 1])
    ll = 0.0; grad = np.zeros_like(beta)
    for t in ev_times:
        d = (time == t) & (event == 1); r = time >= t
        s0 = w[r].sum(); s1 = (w[r, None] * x[r]).sum(0); nd = d.sum()
        ll += eta[d].sum() - nd * np.log(s0)
        grad += x[d].sum(0) - nd * s1 / s0
    return -(ll - 0.5 * alpha * beta @ beta), -(grad - alpha * beta)


def fit(x, time, event, alpha=0.0, maxiter=500):
    res = minimize(lambda b: neg_loglik_grad(b, x, time, event, alpha), np.zeros(x.shape[1]), jac=True,
                   method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-9})
    return res.x


def breslow_cumhaz(time, event, lin_pred):
    """Returns (event_times, H0(t)) using training lin_pred (log relative risk)."""
    w = np.exp(np.clip(lin_pred, -30, 30)); ts = np.unique(time[event == 1]); h = 0.0; out = []
    for t in ts:
        h += ((time == t) & (event == 1)).sum() / w[time >= t].sum(); out.append(h)
    return ts, np.array(out)


def survival_at(ts, H0, lin_pred, grid):
    idx = np.searchsorted(ts, np.asarray(grid, float), side="right") - 1
    h = np.where(idx >= 0, H0[np.clip(idx, 0, None)], 0.0)
    return np.exp(-np.outer(np.exp(np.clip(lin_pred, -30, 30)), h))


# ---- fast vectorised Breslow version (same likelihood as neg_loglik_grad; O(n p)) ----
def prep(x, time, event):
    """Sort once by descending time; tied subjects share the risk set ending at the last member of their tie group."""
    order = np.argsort(-np.asarray(time, float), kind="stable")
    ts = np.asarray(time, float)[order]
    end = np.searchsorted(-ts, -ts, side="right") - 1
    return dict(x=np.asarray(x, float)[order], e=np.asarray(event, int)[order].astype(bool), end=end)


def neg_loglik_grad_fast(beta, P, alpha=0.0):
    x, e, end = P["x"], P["e"], P["end"]
    eta = np.clip(x @ beta, -30, 30)
    w = np.exp(eta)
    s0 = np.cumsum(w)[end]
    s1 = np.cumsum(w[:, None] * x, axis=0)[end]
    ll = eta[e].sum() - np.log(s0[e]).sum() - 0.5 * alpha * beta @ beta
    grad = x[e].sum(0) - (s1[e] / s0[e, None]).sum(0) - alpha * beta
    return -float(ll), -grad


def fit_fast(x, time, event, alpha=0.0, maxiter=500):
    P = prep(x, time, event)
    res = minimize(lambda b: neg_loglik_grad_fast(b, P, alpha), np.zeros(x.shape[1]), jac=True,
                   method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-9})
    return res.x
