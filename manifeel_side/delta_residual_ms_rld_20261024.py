"""rldag (new file, 2026-10-24; multisocket_rldag_20261024): copy of delta_residual_ms_rl_20261023.py + mode 'dagger' (DAgger
round 2): the loaded residual net is EXECUTED and the privileged collect teacher is computed with the same inputs and only LOGGED as the
label (delta_teacher). Every mode now also logs delta_teacher (teacher label; collect: = executed, deploy: zeros) and delta_exec
(= executed delta, = 'delta'). Deploy accepts variants RL and RLD (same ResMLP input layout). 'collect' / 'deploy' computations are
unchanged from the rl file. rl docstring follows.
rlres (new file, 2026-10-23; multisocket_rlres_20261023): copy of delta_residual_ms_rg_20261023.py with ONE change -- the image
block is the served LDP's own frozen obs-encoder features (obs_features, 2078 = what the policy itself perceives, trained only by
behaviour cloning on demos, no geometric label) returned by policy_server_selrule_rl_20261023 for the current request, instead of
ImageNet ResNet18 features. The bridge sets self.feats_src (the PolicyClient); its last_feats must be (n, 2078). Variant 'RL' =
low-dim 100 + obs_features 2078 + raw ff 840. Motivation: multisocket_rgres_20261023 RG_HARM (xy alignment lost; offset probe
GAP_LARGE). rg docstring follows.

rgres (new file, 2026-10-23; multisocket_rgres_20261023): copy of delta_residual_ms_rotres_20261020.py with the task-specific
deploy inputs removed. (1) Image features: frozen ImageNet-pretrained torchvision ResNet18 (resnet18-f37072fd.pth, fc = Identity,
512 per camera, wrist + front = 1024, same ImageNet normalisation and uint8 rounding as before) instead of the ResNet18 of
est_model_ms.pt that was fine-tuned on the socket-offset label. (2) The low-dim image offset estimate (mm/5, 2) is dropped:
low-dim 100 = state 7, EE motion over 1 / 12 steps (6), contact_8d 8, wrist wrench of the last 12 steps (72), base action 6,
steps since contact-event start (1). Unchanged: the privileged collect teacher (training data only), the contact gate
(-Fz > 1 N in the HAND frame = axial force on the tool, an embodiment convention, not the world insertion axis; the axis-free
|F| gate was checked and discriminates true contact worse in this sim: AUROC 0.64 vs 0.80, 8 B16 runs), RHO, ResMLP, raw ff.
Variant 'RG' = low-dim 100 + image 1024 + ff 840. est_path is accepted and ignored (call compatibility). rotres docstring follows.

rotres (new file, 2026-10-20; multisocket_rotres_20261020): copy of delta_residual_ms_v2_20261019.py with three changes --
(1) the collect teacher also corrects plug orientation (rule of delta_residual_ms_rot_20261020: rotation delta =
clip(0.5 * plug_orientation_error(plug, nearest slot) / 0.05 (norm cap 0.3) - a_b_rot, +-RHO) when |err| > 0.5 deg, same active
steps); (2) the raw right tactile force field is recorded / used: ff = [field now (420), field now - field after reset (420)]
(FF = 840, float16 in the log); (3) deploy takes a variant: 'RT' = low-dim + image + ff, 'RC' = low-dim + image (v2 inputs, the
c8d summary stays inside low-dim). ff_perm (deploy only) feeds env i the ff of env perm[i] (tactile-shuffle ablation).
A correction row is one where the xy or the rotation teacher term is active. v2 docstring follows.
v2 (new file, 2026-10-19; multisocket_dagger_v2_20261019): copy of delta_residual_ms_20261019.py with ONE change -- the residual
(teacher in collect, network in deploy) is active only while in contact NOW (measured -Fz > 1 N within the last ACTIVE_GAP = 3 steps)
instead of from the first contact latch to the end of the episode, so it no longer pulls toward the slot being left during a
switch transit (v1: p_other 0.40 -> 0.15). The low-dim slot "steps since contact latch" becomes "steps since the current contact
event began" (event = contact after > ACTIVE_GAP contact-free steps). v1 docstring follows.
Lower-level residual for the multisocket hierarchy, CR-DAgger style (new file, 2026-10-19; multisocket_dagger_20261019).

The frozen base (MS5F: LDP + AT re-decoding) acts every step; after a DEPLOYABLE contact latch (measured wrist wrench
-Fz > 1 N) a residual delta is added: u = clip(a_b + delta, -1, 1), |delta| <= RHO.
  collect  on-policy delta corrections by a privileged teacher (training data only): while the plug bottom is more than
           TOL_M from the NEAREST slot centre in xy, delta_xy = clip(KP * (slot - plug)_xy / POS_SCALE - a_b_xy, +-RHO), other
           dims 0 (the base keeps inserting); aligned -> delta 0. The same rule on both sockets: the label depends only on the
           offset to the nearest slot, never on which socket fits.
  deploy   delta = RHO * tanh(MLP(x)) from the trained residual (residual_ms.pt).
Residual input x (all deployable): low-dim 102 = state 7, EE motion over 1 / 12 steps (x100, 6), contact_8d 8, wrist wrench
of the last 12 steps (/10, 72), base action 6, steps since contact latch /100, image offset estimate (mm/5, 2); plus the
1024-d frozen ResNet18 features of est_model_ms.pt (wrist + front). Recorded per step to delta_log.npz."""
import numpy as np
import torch
import torch.nn as nn

import obs_corrector as oc
from dagger_expert import plug_orientation_error

RHO, TOL_M, KP, POS_SCALE, CONTACT_FZ, WIN = 0.2, 0.0005, 0.6, 0.01, 1.0, 12
ACTIVE_GAP = 3
LD, IMG, ACT, FF = 100, 2078, 6, 840
RESNET18_IMAGENET = '/home/dyros-recruit/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
KP_ROT, ROT_SCALE, ROT_CAP, ROT_TOL = 0.5, 0.05, 0.3, np.radians(0.5)


class GenericEncoder:
    """frozen ImageNet ResNet18 trunk (no task label), same interface as obs_corrector.Estimator (r, m, s, dev, torch)"""
    def __init__(self):
        import torchvision
        r = torchvision.models.resnet18(); r.load_state_dict(torch.load(RESNET18_IMAGENET, map_location='cpu')); r.fc = nn.Identity()
        self.dev = 'cuda:0'; self.torch = torch; self.r = r.to(self.dev).eval()
        self.m = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1); self.s = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)


class ResMLP(nn.Module):
    def __init__(self, d=LD + IMG, h=256):
        super().__init__()
        self.register_buffer('mu', torch.zeros(d)); self.register_buffer('sd', torch.ones(d))
        self.net = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(),
                                 nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, ACT))

    def forward(self, x):
        return RHO * torch.tanh(self.net((x - self.mu) / self.sd))


class DeltaResidual:
    def __init__(self, n, mode, est_path, ckpt=None, ff_perm=None):
        assert mode in ('collect', 'deploy', 'dagger')
        self.n, self.mode = n, mode; self.dev = 'cuda:0'; self.t = torch; self.feats_src = None
        self.net = None; self.variant = None; self.ff_perm = ff_perm
        if mode in ('deploy', 'dagger'):
            ck = torch.load(ckpt, map_location=self.dev); self.variant = ck['variant']
            assert self.variant in ('RL', 'RLD'); self.net = ResMLP(d=LD + IMG + FF).to(self.dev).eval(); self.net.load_state_dict(ck['model'])
        self.latch = np.full(n, -1); self.alive = np.ones(n, bool); self.ee_hist = []; self.w_hist = []
        self.last_con = np.full(n, -10 ** 6); self.ev_on = np.full(n, -1)
        self.prev_corr = np.zeros(n, bool)
        self.buf = {k: [] for k in ('ld', 'img', 'ff', 'ab', 'delta', 'rec', 'corr', 'corr_start', 'off_true', 'rot_err_deg', 'delta_teacher', 'delta_exec')}

    def step(self, action, obs, c8, wrench_hist, success_hist, plug_xy, slot_xy, ff_now, ff_init, plug_quat=None, slot_quat=None):
        """wrench_hist: list of (n,6) measured wrist wrench after each step so far; ff_now / ff_init: (n,3,10,14) raw right tactile
        force field (live / after reset); plug_xy / slot_xy / plug_quat / slot_quat (n,2,4) privileged (collect only)."""
        n, t = self.n, self.t; ab = np.asarray(action, np.float32).copy(); st = np.asarray(obs['state'], np.float32)
        self.ee_hist.append(st[:, :3].copy()); k = len(wrench_hist)
        if success_hist:
            self.alive &= ~(np.asarray(success_hist[-1]) > 0)
        if k:
            con = -np.asarray(wrench_hist[-1])[:, 2] > CONTACT_FZ
            new = (self.latch < 0) & con; self.latch[new] = k
            start = con & (k - self.last_con > ACTIVE_GAP); self.ev_on[start] = k; self.last_con[con] = k
        act = (self.latch >= 0) & self.alive & (k - self.last_con <= ACTIVE_GAP)
        ld = np.zeros((n, LD), np.float32); img = np.zeros((n, IMG), np.float16); delta = np.zeros((n, ACT), np.float32)
        ffr = np.zeros((n, FF), np.float16); red = np.full(n, np.nan, np.float32)
        corr = np.zeros(n, bool); off_true = np.full((n, 2), np.nan, np.float32); tdel = np.zeros((n, ACT), np.float32)
        if act.any():
            idx = np.where(act)[0]
            fs_ = None if self.feats_src is None else self.feats_src.last_feats
            if fs_ is None or np.asarray(fs_).shape != (n, IMG):
                raise RuntimeError(f'RL residual needs per-step obs_features (n, {IMG}) from the server (want_feats)')
            fi = np.asarray(fs_, np.float32)[idx]
            wh = np.stack([np.asarray(wrench_hist[max(0, k - WIN + j)], np.float32) for j in range(WIN)], 1)[idx]
            h1, h12 = self.ee_hist[max(0, len(self.ee_hist) - 2)][idx], self.ee_hist[max(0, len(self.ee_hist) - 13)][idx]
            s = st[idx]
            ld[idx] = np.concatenate([s, (s[:, :3] - h1) * 100, (s[:, :3] - h12) * 100, np.asarray(c8, np.float32)[idx],
                                      wh.reshape(len(idx), -1) / 10.0, ab[idx], ((k - self.ev_on[idx]) / 100.0)[:, None]], 1)
            img[idx] = fi.astype(np.float16)
            fn = np.asarray(ff_now, np.float32).reshape(n, -1); f0 = np.asarray(ff_init, np.float32).reshape(n, -1)
            if self.ff_perm is not None:
                fn, f0 = fn[self.ff_perm], f0[self.ff_perm]
            ffr[idx] = np.concatenate([fn, fn - f0], 1)[idx].astype(np.float16)
            if self.mode in ('collect', 'dagger'):
                near = np.argmin(np.linalg.norm(plug_xy[idx, None] - slot_xy[idx], axis=-1), axis=1)
                off = plug_xy[idx] - slot_xy[idx, near]; off_true[idx] = off * 1000
                c = np.linalg.norm(off, axis=1) > TOL_M
                want = np.clip(KP * (-off) / POS_SCALE, -oc.XY_CAP, oc.XY_CAP)
                d = np.clip(want - ab[idx, :2], -RHO, RHO) * c[:, None]
                tdel[idx, :2] = d
                err = plug_orientation_error(np.asarray(plug_quat, np.float32)[idx], np.asarray(slot_quat, np.float32)[idx, near])
                ne = np.linalg.norm(err, axis=-1); red[idx] = np.degrees(ne); cr = ne > ROT_TOL
                wr = KP_ROT * err / ROT_SCALE; wr = wr * np.minimum(1.0, ROT_CAP / np.maximum(np.linalg.norm(wr, axis=-1, keepdims=True), 1e-9))
                tdel[idx, 3:6] = np.clip(wr - ab[idx, 3:6], -RHO, RHO) * cr[:, None]
                corr[idx] = c | cr
            if self.mode in ('deploy', 'dagger'):
                with t.no_grad():
                    parts = [ld[idx], img[idx].astype(np.float32)] + [ffr[idx].astype(np.float32)]
                    x = t.from_numpy(np.concatenate(parts, 1)).to(self.dev)
                    delta[idx] = self.net(x).cpu().numpy()
            else:
                delta[idx] = tdel[idx]
        u = np.clip(ab + delta, -1, 1).astype(np.float32)
        cs = corr & ~self.prev_corr; self.prev_corr = corr.copy()
        for key, v in (('ld', ld), ('img', img), ('ff', ffr), ('ab', ab), ('delta', delta), ('rec', act.copy()), ('corr', corr), ('corr_start', cs), ('off_true', off_true), ('rot_err_deg', red), ('delta_teacher', tdel), ('delta_exec', delta.copy())):
            self.buf[key].append(v)
        return u

    def save(self, path, success):
        d = {k: np.stack(v) for k, v in self.buf.items()}
        d['success'] = np.asarray(success)[:len(d['rec'])] > 0; d['latch'] = self.latch; d['mode'] = self.mode
        np.savez_compressed(path, **d)
