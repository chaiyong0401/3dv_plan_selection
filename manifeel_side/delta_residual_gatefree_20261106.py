"""gatefree (new file, 2026-11-06; usb_keyed_s12r_gatefree_res_20261106): copy of delta_residual_ms_rld_20261024.py (unchanged
original) with three changes (criteria.json residual_module):
  1. act = alive: the residual (teacher in collect, network in deploy) is evaluated at EVERY step until success; the contact
     latch / event bookkeeping is kept only to fill the 'steps since contact event' input (same low-dim layout LD 100).
  2. teacher zone (collect only, privileged): the teacher corrects only when plug-socket height <= ZONE_H_MM and the xy offset to
     the nearest slot <= ZONE_XY_MM; outside the zone delta_teacher = 0 (so the executed delta is 0 there in collect).
  3. teacher rotation only when the error to the nearer 180-deg orientation is <= ROT_GATE_DEG (and inside the zone).
Recording: every zone row; outside the zone every FAR_EVERY-th step per env (sample_w = FAR_EVERY). Logged per row: zone,
sample_w, height_mm, xy_mm, rot_err_deg (privileged, logging/analysis only in deploy). Deploy: network delta added at every step.
Unchanged: xy / rot teacher rules, RHO, ResMLP, inputs (low-dim 100 + obs_features 2078 + ff 840). robot_command_sent: false."""
import numpy as np
import torch

import obs_corrector as oc
from dagger_expert import plug_orientation_error
from delta_residual_ms_rld_20261024 import ResMLP, RHO, TOL_M, KP, POS_SCALE, CONTACT_FZ, WIN, ACTIVE_GAP, LD, IMG, ACT, FF, \
    KP_ROT, ROT_SCALE, ROT_CAP, ROT_TOL

ZONE_H_MM, ZONE_XY_MM, ROT_GATE_DEG, FAR_EVERY = 30.0, 10.0, 15.0, 6


class DeltaResidualGF:
    def __init__(self, n, mode, ckpt=None):
        assert mode in ('collect', 'deploy')
        self.n, self.mode = n, mode; self.dev = 'cuda:0'; self.t = torch; self.feats_src = None; self.net = None
        if mode == 'deploy':
            ck = torch.load(ckpt, map_location=self.dev); assert ck['variant'] in ('RL', 'GF'), ck['variant']
            self.net = ResMLP(d=LD + IMG + FF).to(self.dev).eval(); self.net.load_state_dict(ck['model'])
        self.latch = np.full(n, -1); self.alive = np.ones(n, bool); self.ee_hist = []
        self.last_con = np.full(n, -10 ** 6); self.ev_on = np.full(n, -1); self.prev_corr = np.zeros(n, bool)
        self.rows = {k: [] for k in ('t', 'e', 'ld', 'img', 'ff', 'ab', 'delta', 'delta_teacher', 'corr', 'corr_start', 'zone', 'sample_w',
                                     'height_mm', 'xy_mm', 'rot_err_deg', 'off_true')}
        self.k_step = 0

    def step(self, action, obs, c8, wrench_hist, success_hist, plug_pos, slot_pos, ff_now, ff_init, plug_quat, slot_quat):
        """plug_pos (n,3), slot_pos (n,S,3), plug_quat (n,4), slot_quat (n,S,4): privileged (teacher in collect; logging in deploy)."""
        n = self.n; ab = np.asarray(action, np.float32).copy(); st = np.asarray(obs['state'], np.float32)
        self.ee_hist.append(st[:, :3].copy()); k = len(wrench_hist)
        if success_hist:
            self.alive &= ~(np.asarray(success_hist[-1]) > 0)
        if k:
            con = -np.asarray(wrench_hist[-1])[:, 2] > CONTACT_FZ
            new = (self.latch < 0) & con; self.latch[new] = k
            start = con & (k - self.last_con > ACTIVE_GAP); self.ev_on[start] = k; self.last_con[con] = k
        act = self.alive.copy()                                                     # change 1: no contact gate
        delta = np.zeros((n, ACT), np.float32)
        if act.any():
            idx = np.where(act)[0]
            fs_ = None if self.feats_src is None else self.feats_src.last_feats
            if fs_ is None or np.asarray(fs_).shape != (n, IMG):
                raise RuntimeError(f'residual needs per-step obs_features (n, {IMG}) from the server (want_feats)')
            fi = np.asarray(fs_, np.float32)[idx]
            wh = np.stack([np.asarray(wrench_hist[max(0, k - WIN + j)], np.float32) if k else np.zeros((n, 6), np.float32)
                           for j in range(WIN)], 1)[idx]
            h1, h12 = self.ee_hist[max(0, len(self.ee_hist) - 2)][idx], self.ee_hist[max(0, len(self.ee_hist) - 13)][idx]
            s = st[idx]
            ld = np.concatenate([s, (s[:, :3] - h1) * 100, (s[:, :3] - h12) * 100, np.asarray(c8, np.float32)[idx],
                                 wh.reshape(len(idx), -1) / 10.0, ab[idx], ((k - self.ev_on[idx]) / 100.0)[:, None]], 1).astype(np.float32)
            fn = np.asarray(ff_now, np.float32).reshape(n, -1); f0 = np.asarray(ff_init, np.float32).reshape(n, -1)
            ffr = np.concatenate([fn, fn - f0], 1)[idx]
            # privileged geometry (teacher in collect; logging only in deploy)
            pp, sp = np.asarray(plug_pos, np.float32)[idx], np.asarray(slot_pos, np.float32)[idx]
            near = np.argmin(np.linalg.norm(pp[:, None, :2] - sp[:, :, :2], axis=-1), axis=1); sl = sp[np.arange(len(idx)), near]
            off = pp[:, :2] - sl[:, :2]; xy_mm = np.linalg.norm(off, axis=1) * 1000; h_mm = (pp[:, 2] - sl[:, 2]) * 1000
            err = plug_orientation_error(np.asarray(plug_quat, np.float32)[idx], np.asarray(slot_quat, np.float32)[idx, near])
            ne = np.linalg.norm(err, axis=-1); red = np.degrees(ne)
            zone = (h_mm <= ZONE_H_MM) & (xy_mm <= ZONE_XY_MM)                    # change 2
            tdel = np.zeros((len(idx), ACT), np.float32)
            c = (np.linalg.norm(off, axis=1) > TOL_M) & zone
            want = np.clip(KP * (-off) / POS_SCALE, -oc.XY_CAP, oc.XY_CAP)
            tdel[:, :2] = np.clip(want - ab[idx, :2], -RHO, RHO) * c[:, None]
            cr = (ne > ROT_TOL) & (red <= ROT_GATE_DEG) & zone                     # change 3
            wr = KP_ROT * err / ROT_SCALE; wr = wr * np.minimum(1.0, ROT_CAP / np.maximum(np.linalg.norm(wr, axis=-1, keepdims=True), 1e-9))
            tdel[:, 3:6] = np.clip(wr - ab[idx, 3:6], -RHO, RHO) * cr[:, None]
            corr = c | cr
            if self.mode == 'deploy':
                with torch.no_grad():
                    x = torch.from_numpy(np.concatenate([ld, fi, ffr], 1)).to(self.dev)
                    delta[idx] = self.net(x).cpu().numpy()
            else:
                delta[idx] = tdel
            cprev = self.prev_corr[idx]; cs = corr & ~cprev; self.prev_corr[:] = False; self.prev_corr[idx] = corr
            keep = zone | (k % FAR_EVERY == 0)
            sw = np.where(zone, 1.0, float(FAR_EVERY)).astype(np.float32)
            for key, v in (('t', np.full(len(idx), k)), ('e', idx), ('ld', ld), ('img', fi.astype(np.float16)), ('ff', ffr.astype(np.float16)),
                           ('ab', ab[idx]), ('delta', delta[idx]), ('delta_teacher', tdel), ('corr', corr), ('corr_start', cs), ('zone', zone),
                           ('sample_w', sw), ('height_mm', h_mm), ('xy_mm', xy_mm), ('rot_err_deg', red), ('off_true', off * 1000)):
                self.rows[key].append(np.asarray(v)[keep])
        self.k_step += 1
        return np.clip(ab + delta, -1, 1).astype(np.float32)

    def save(self, path, success):
        d = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in self.rows.items()}
        d['success'] = np.asarray(success) > 0; d['mode'] = self.mode
        d['consts'] = np.array([ZONE_H_MM, ZONE_XY_MM, ROT_GATE_DEG, FAR_EVERY], np.float32)
        np.savez_compressed(path, **d)


if __name__ == '__main__':   # unit checks (no GPU, no network): zone / rot gate / recording / bound
    class FS: last_feats = np.zeros((3, IMG), np.float16)
    r = DeltaResidualGF(3, 'collect'); r.feats_src = FS()
    q0 = np.array([0, 0, 0, 1], np.float32); rz = lambda d: np.array([0, 0, np.sin(np.radians(d) / 2), np.cos(np.radians(d) / 2)], np.float32)
    plug = np.array([[0.004, 0, 0.020], [0.004, 0, 0.060], [0.004, 0, 0.020]], np.float32)       # in zone / above zone / in zone
    slot = np.array([[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]], np.float32)
    pq = np.stack([rz(5), rz(5), rz(60)]); sq = np.stack([q0, q0, q0])[:, None]
    obs = {'state': np.zeros((3, 7), np.float32)}
    wh = [np.zeros((3, 6), np.float32)]
    for kk in range(7):
        u = r.step(np.zeros((3, 6)), obs, np.zeros((3, 8)), wh * (kk + 1), [], plug, slot, np.zeros((3, 3, 10, 14)), np.zeros((3, 3, 10, 14)), pq, sq)
    d = {k: np.concatenate(v) for k, v in r.rows.items()}
    z0 = d['e'] == 0; z1 = d['e'] == 1; z2 = d['e'] == 2
    assert d['zone'][z0].all() and not d['zone'][z1].any() and d['zone'][z2].all()
    assert (np.abs(d['delta_teacher'][z0, :2]).sum(1) > 0).all() and (np.abs(d['delta_teacher'][z0, 3:]).sum(1) > 0).all()   # xy + rot (5 deg)
    assert (d['delta_teacher'][z1] == 0).all()                                                                          # outside zone
    assert (np.abs(d['delta_teacher'][z2, :2]).sum(1) > 0).all() and (d['delta_teacher'][z2, 3:] == 0).all()            # 60 deg: xy only
    assert z1.sum() == 1 and (d["sample_w"][z1] == FAR_EVERY).all() and z0.sum() == 7                                   # k = 1..7: far row only at k = 6
    assert np.abs(d['delta']).max() <= RHO + 1e-6
    print('gatefree selftest OK', int(z1.sum()), 'far rows')
