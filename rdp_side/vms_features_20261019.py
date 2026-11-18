"""Multisocket verifier (VMS) features, new file 2026-10-19 (multisocket_verifier_train_20261019).

One numpy-only implementation used both offline (training: replay of rollout_traj.npz) and online (the bridge
eval_rdp_bridge_multisocket_vms_20261019.py sends the features with every request), so train and deploy cannot drift.
Inputs per control step are DEPLOYABLE only: measured wrist wrench (F/T), EE position (proprioception), live contact_8d.

VMSHistory.update(k, wrench, ee, c8d) is called with the state AFTER action k (= rollout_traj index k);
VMSHistory.features() then describes the history up to and including k, i.e. what is known when the request for
step k + 1 is built.  H (14) and T (16) are returned concatenated as (N, 30).
"""
import collections

import numpy as np

CONTACT_FZ = 1.0          # N, -Fz_meas > this = contact (same trigger as the estimator studies)
ONSET_GAP = 6             # contact-free steps before a new contact event
REGION_MM = 15.0          # EE xy move from the region anchor that starts a new region
STALL_WIN, STALL_MM = 4, 0.5   # stall = contact and EE z descended < 0.5 mm over 4 steps
WIN = 12
H_DIM, T_DIM = 14, 16
CAND_DIM = 13


class VMSHistory:
    def __init__(self, n):
        self.n = n
        self.k = -1
        self.last_contact = np.full(n, -10 ** 6)
        self.onset = np.full(n, -1)
        self.n_events = np.zeros(n)
        self.z_onset = np.zeros(n)
        self.anchor = None
        self.region_changes = np.zeros(n)
        self.stalls = np.zeros(n)
        self.prev_stall = np.zeros(n, bool)
        self.contact = np.zeros(n, bool)
        self.ee = np.zeros((n, 3))
        self.z_hist = collections.deque(maxlen=STALL_WIN + 1)
        self.w_hist = collections.deque(maxlen=WIN)
        self.c_hist = collections.deque(maxlen=WIN)

    def update(self, k, wrench, ee, c8d):
        wrench = np.asarray(wrench, np.float64); ee = np.asarray(ee, np.float64); c8d = np.asarray(c8d, np.float64)
        self.k = k; self.ee = ee
        if self.anchor is None:
            self.anchor = ee[:, :2].copy()
        moved = np.linalg.norm(ee[:, :2] - self.anchor, axis=1) * 1000 > REGION_MM
        self.anchor[moved] = ee[moved, :2]; self.region_changes[moved] += 1; self.stalls[moved] = 0
        con = -wrench[:, 2] > CONTACT_FZ
        start = con & (k - self.last_contact > ONSET_GAP)
        self.onset[start] = k; self.n_events[start] += 1; self.z_onset[start] = ee[start, 2]
        self.last_contact[con] = k; self.contact = con
        self.z_hist.append(ee[:, 2])
        if len(self.z_hist) == STALL_WIN + 1:
            stall = con & ((self.z_hist[0] - self.z_hist[-1]) * 1000 < STALL_MM)
        else:
            stall = np.zeros(self.n, bool)
        self.stalls[stall & ~self.prev_stall] += 1; self.prev_stall = stall
        self.w_hist.append(wrench); self.c_hist.append(c8d)

    def features(self):
        k = self.k
        in_event = (self.onset >= 0) & (k - self.last_contact <= ONSET_GAP)
        since = np.where(in_event, (k - self.onset) / 100.0, 0.0)
        dz = np.where(in_event, (self.ee[:, 2] - self.z_onset) * 1000 / 10.0, 0.0)
        W = np.stack(self.w_hist, 0); fxy = np.linalg.norm(W[..., :2], axis=-1)
        H = np.column_stack([np.full(self.n, k / 500.0), self.contact.astype(float), since, self.n_events / 6.0, self.stalls / 4.0,
                             (self.ee[:, :2] - self.anchor) * 1000 / 50.0, dz, self.region_changes / 3.0,
                             W[..., 2].mean(0) / 10.0, W[..., 2].min(0) / 10.0, fxy.mean(0) / 10.0, fxy.max(0) / 10.0,
                             (-W[..., 2] > CONTACT_FZ).mean(0)])
        T = np.column_stack([self.c_hist[-1], np.stack(self.c_hist, 0).mean(0)])
        out = np.concatenate([H, T], 1).astype(np.float32)
        assert out.shape == (self.n, H_DIM + T_DIM)
        return out


def cand_features(plan_mean, lateral, dist, plan_raw):
    """(k, N, 13): verifier v3's 12 candidate features (normalized plan mean 6, lateral, dist, valid, unit mean planned
    raw xy 2, its norm) + mean planned raw z.  Works on numpy or torch arrays (server passes torch)."""
    lib = np if isinstance(plan_mean, np.ndarray) else __import__('torch')
    raw_xy = plan_raw[..., :2].mean(-2); raw_z = plan_raw[..., 2].mean(-1)
    xyn = (raw_xy ** 2).sum(-1, keepdims=True) ** 0.5 if lib is np else raw_xy.norm(dim=-1, keepdim=True)
    ones = lib.ones_like(dist[..., None])
    cat = (lambda xs: np.concatenate(xs, -1)) if lib is np else (lambda xs: lib.cat(xs, -1))
    return cat([plan_mean, lateral[..., None], dist[..., None], ones, raw_xy / (xyn + 1e-9), xyn, raw_z[..., None]])


def replay(traj, steps):
    """Offline: rollout_traj dict -> features at each request step s in `steps` (history up to traj index s-1).
    Returns (len(steps), N, 30); a step with s == 0 gets NaN."""
    ww, ee, c8 = traj['wrist_wrench'], traj['ee_pos'], traj['contact_8d_right']
    n = ww.shape[1]; h = VMSHistory(n); out = {}; want = set(int(s) for s in steps)
    if 0 in want:
        out[0] = np.full((n, H_DIM + T_DIM), np.nan, np.float32)
    for k in range(ww.shape[0]):
        h.update(k, ww[k], ee[k], c8[k])
        if k + 1 in want:
            out[k + 1] = h.features()
    return np.stack([out[int(s)] for s in steps], 0)
