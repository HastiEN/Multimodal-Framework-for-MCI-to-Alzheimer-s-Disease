"""C3: corrected IPCW Brier score / IBS and Kaplan-Meier null benchmark.

Differences from model_survival.ipcw_brier_score (historical, kept untouched):
  * the mean is taken over ALL test subjects (denominator n), as in Graf et al.;
    subjects censored before t contribute weight 0 but still count in n.
  * G(t) is the training-fold KM estimate of the censoring distribution, evaluated at
    T_i- (left limit) for events, per Graf; implemented with a right-continuous step
    function evaluated at T_i, matching scikit-survival for tie-free toy data.
  * IBS divides the trapezoid integral by the grid length.
"""
import numpy as np


def _km_step(time, event):
    """Return (times, S) of the KM estimator for `event` as the event indicator."""
    time = np.asarray(time, float); event = np.asarray(event, int)
    ut = np.unique(time[event == 1]); s = 1.0; out = []
    for t in ut:
        n = np.sum(time >= t); d = np.sum((time == t) & (event == 1))
        s *= 1 - d / n; out.append(s)
    return ut, np.array(out)


def km_at(q, time, event):
    ut, s = _km_step(time, event)
    q = np.atleast_1d(np.asarray(q, float))
    idx = np.searchsorted(ut, q, side="right") - 1
    return np.where(idx >= 0, s[np.clip(idx, 0, None)] if len(s) else 1.0, 1.0)


def km_at_left(q, time, event):
    """KM evaluated at t- (strictly before t)."""
    ut, s = _km_step(time, event)
    q = np.atleast_1d(np.asarray(q, float))
    idx = np.searchsorted(ut, q, side="left") - 1
    return np.where(idx >= 0, s[np.clip(idx, 0, None)] if len(s) else 1.0, 1.0)


def brier_ipcw(train_time, train_event, test_time, test_event, surv_prob_at_t, t, floor=1e-3):
    """Corrected IPCW Brier score at horizon t. surv_prob_at_t: predicted S(t|x) for test subjects."""
    tt = np.asarray(test_time, float); te = np.asarray(test_event, int); p = np.asarray(surv_prob_at_t, float)
    cens_train = 1 - np.asarray(train_event, int)
    g_t = max(float(km_at(t, train_time, cens_train)[0]), floor)
    g_i = np.maximum(km_at(tt, train_time, cens_train), floor)
    ev = (tt <= t) & (te == 1)
    surv = tt > t
    loss = np.zeros(len(tt))
    loss[ev] = (p[ev] ** 2) / g_i[ev]
    loss[surv] = ((1 - p[surv]) ** 2) / g_t
    return float(loss.mean())  # denominator = all test subjects


def integrated_brier(times, scores):
    times = np.asarray(times, float); scores = np.asarray(scores, float)
    return float(np.trapezoid(scores, times) / (times[-1] - times[0]))


def km_null_survival(train_time, train_event, grid):
    """Predicted S(t) = training-fold KM, identical for every test subject."""
    return km_at(grid, train_time, train_event)
