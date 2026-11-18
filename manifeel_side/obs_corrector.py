"""obs_corrector_20261015: one-shot contact corrector for the PiH bridge (criteria.json of that study).

Trigger: 12 steps after contact onset (first |plug_socket_force| > 3 N, the rule of residual_img_probe_20261015), once per
env, skipped if the ee already went down > 10 mm since onset (inserting) or |offset| < 1 mm. No sim success flag is used. offset = plug_xy - socket_xy: oracle reads the sim, est
predicts it from the wrist+front frames of that same step with the image estimator. Procedure (no rotation, no noise):
LIFT ee z +5 mm holding xy -> MOVE ee xy to ee_xy(trigger) - offset at that height -> DESCEND (z action -0.29, xy held)
until ee z <= ee_z(trigger) - 24 mm or 80 steps -> hand back to the policy. Gains/caps from dagger_expert.PARAMS."""
import numpy as np

POS_SCALE, KP_XY, XY_CAP, Z_CAP, DESC_Z = 0.01, 0.6, 0.5, 0.5, -0.29
LIFT_M, MOVE_TOL_M, MAX_LIFT, MAX_MOVE, MAX_DESC, DESC_M, MIN_OFF_M = 0.005, 0.0005, 20, 60, 80, 0.024, 0.001


class Estimator:
    def __init__(self, path):
        import torch, torch.nn as nn, torchvision
        ck = torch.load(path, map_location='cpu'); r = torchvision.models.resnet18(); r.fc = nn.Identity()
        self.r, self.h = r, nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 2))
        self.r.load_state_dict(ck['r']); self.h.load_state_dict(ck['h'])
        self.dev = 'cuda:0'; self.r.to(self.dev).eval(); self.h.to(self.dev).eval(); self.torch = torch
        self.ym, self.ys = np.asarray(ck['ym']), np.asarray(ck['ys'])
        self.m = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1); self.s = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)

    def __call__(self, wrist, front):  # (n,3,H,W) float [0,1] -> offset mm (n,2)
        t = self.torch
        with t.no_grad():
            u8 = lambda v: t.from_numpy((np.clip(v, 0, 1) * 255).round().astype(np.uint8)).to(self.dev).float() / 255
            x = (t.stack([u8(wrist), u8(front)], 1).flatten(0, 1) - self.m) / self.s
            f = self.r(x).view(len(wrist), -1); y = self.h(f).cpu().numpy()
        return y * self.ys + self.ym


class ContactCorrector:
    def __init__(self, n, mode, est_path):
        self.n, self.mode = n, mode; self.est = Estimator(est_path) if mode == 'est' else None
        self.onset = np.full(n, -1); self.phase = np.zeros(n, int)  # 0 idle, 1 lift, 2 move, 3 descend, 4 done
        self.k = np.zeros(n, int); self.ok = np.zeros(n, int); self.target = np.zeros((n, 3)); self.z0 = np.zeros(n)
        self.t_trig = np.full(n, -1); self.off_used = np.full((n, 2), np.nan); self.off_true = np.full((n, 2), np.nan)
        self.skipped = np.zeros(n, int); self.z_on = np.zeros(n)  # skipped: 1 inserting (ee dz < -10 mm since onset), 2 |offset| < 1 mm
        self.mode_log = []

    def step(self, action, obs, traj, task):
        if not traj['plug_socket_force']:
            self.mode_log.append(self.phase.copy()); return action
        k = len(traj['plug_socket_force']) - 1; f = np.linalg.norm(traj['plug_socket_force'][-1], axis=-1)
        ee = np.asarray(obs['state'])[:, :3].astype(np.float64)
        new = (self.onset < 0) & (f > 3.0); self.onset[new] = k; self.z_on[new] = ee[new, 2]
        trig = (self.phase == 0) & (self.onset >= 0) & (self.onset + 12 == k)
        if trig.any():
            idx = np.where(trig)[0]
            true = (np.asarray(traj['plug_pos'][-1])[idx, :2] - np.asarray(traj['socket_pos'][-1])[idx, :2])
            off = true if self.mode == 'oracle' else self.est(np.asarray(obs['wrist'])[idx], np.asarray(obs['front'])[idx]) / 1000.0
            for j, e in enumerate(idx):
                self.off_true[e] = true[j]; self.off_used[e] = off[j]; self.t_trig[e] = k
                if ee[e, 2] < self.z_on[e] - 0.010:
                    self.phase[e], self.skipped[e] = 4, 1
                elif np.linalg.norm(off[j]) < MIN_OFF_M:
                    self.phase[e], self.skipped[e] = 4, 2
                else:
                    self.phase[e], self.k[e] = 1, 0; self.z0[e] = ee[e, 2]
                    self.target[e] = [ee[e, 0] - off[j][0], ee[e, 1] - off[j][1], ee[e, 2] + LIFT_M]
        act = np.array(action, dtype=np.float32, copy=True)
        for e in np.where((self.phase >= 1) & (self.phase <= 3))[0]:
            a = np.zeros(6, np.float32); self.k[e] += 1
            if self.phase[e] == 1:
                a[:2] = 0.0; a[2] = np.clip((self.target[e, 2] - ee[e, 2]) / POS_SCALE, -Z_CAP, Z_CAP)
                if abs(self.target[e, 2] - ee[e, 2]) < 0.001 or self.k[e] >= MAX_LIFT:
                    self.phase[e], self.k[e] = 2, 0
            elif self.phase[e] == 2:
                d = self.target[e, :2] - ee[e, :2]
                a[:2] = np.clip(KP_XY * d / POS_SCALE, -XY_CAP, XY_CAP); a[2] = np.clip((self.target[e, 2] - ee[e, 2]) / POS_SCALE, -Z_CAP, Z_CAP)
                self.ok[e] = self.ok[e] + 1 if np.linalg.norm(d) < MOVE_TOL_M else 0
                if self.ok[e] >= 3 or self.k[e] >= MAX_MOVE:
                    self.phase[e], self.k[e] = 3, 0
            else:
                d = self.target[e, :2] - ee[e, :2]
                a[:2] = np.clip(KP_XY * d / POS_SCALE, -XY_CAP, XY_CAP); a[2] = DESC_Z
                if ee[e, 2] <= self.z0[e] - DESC_M or self.k[e] >= MAX_DESC:
                    self.phase[e] = 4
            act[e] = a
        self.mode_log.append(self.phase.copy())
        return act

    def save(self, path):
        np.savez_compressed(path, t_trig=self.t_trig, onset=self.onset, off_used=self.off_used, off_true=self.off_true,
                            skipped=self.skipped, mode_log=np.stack(self.mode_log).astype(np.int8), mode=self.mode)
