"""errsim (new file, 2026-11-18; errsim_20261118): DeltaResidualHFC (delta_residual_conv_20261111, unchanged; the Q400B residual
residual_conv.pt) with the rz output (action dim 5) replaced by an error-predicting head (errsim_20261118/errnet.pt, path in env
RDP_ERRNET): e_hat = signed yaw error to the nearer 180 deg (deg), gate logit g (teacher rotation active).
rz = 1[g > 0] * clip(clip(-0.1745 * e_hat, +-0.3) - generator_rz, +-0.2)  (teacher constants KP_ROT 0.5, ROT_SCALE 0.05, ROT_CAP 0.3).
Other outputs, inputs and logging unchanged. e_hat / g per step saved to es_log.npz. robot_command_sent: false."""
import os, sys, numpy as np, torch
from delta_residual_conv_20261111 import DeltaResidualHFC, HandFrameConv
from delta_residual_handframe_20261106 import XY_SLOTS
sys.path.insert(0, '/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs/errsim_20261118')
from train_err import ErrNet
AB_RZ = 98   # ld = state 7 | dEE1 3 | dEE12 3 | c8 8 | wrench 72 | base action 6 (93:99) | event 1


def to_hand_t(x):
    q = x[:, 3:7]; psi = torch.atan2(2 * (q[:, 0] * q[:, 1] + q[:, 3] * q[:, 2]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
    c, s = torch.cos(psi), torch.sin(psi); x = x.clone()
    for i, _ in XY_SLOTS:
        vx, vy = x[:, i].clone(), x[:, i + 1].clone(); x[:, i], x[:, i + 1] = c * vx + s * vy, -s * vx + c * vy
    return x


class HandFrameConvES(HandFrameConv):
    def forward(self, x):
        out = HandFrameConv.forward(self, x).clone()
        eh, g = self.errnet(to_hand_t(x)); ab = x[:, AB_RZ]
        rz = torch.clamp(torch.clamp(-0.1745 * eh, -0.3, 0.3) - ab, -0.2, 0.2)
        out[:, 5] = torch.where(g > 0, rz, torch.zeros_like(rz)); self.es_last = torch.stack([eh, g], 1).detach()
        return out


class DeltaResidualES(DeltaResidualHFC):
    def __init__(self, n, mode, ckpt=None):
        super().__init__(n, mode, ckpt); ck = torch.load(os.environ['RDP_ERRNET'], map_location=self.dev)
        en = ErrNet(ck['d'], np.zeros(ck['d'], np.float32), np.ones(ck['d'], np.float32)).to(self.dev).eval(); en.load_state_dict(ck['model'])
        self.net.__class__ = HandFrameConvES; self.net.errnet = en; self.es_log = []

    def step(self, *a, **k):
        self.net.es_last = None; out = super().step(*a, **k); v = np.full((self.n, 2), np.nan, np.float32)
        if self.net.es_last is not None:
            idx = np.where(self.alive)[0]; v[idx] = self.net.es_last.cpu().numpy()
        self.es_log.append(v); return out

    def save(self, path, success):
        super().save(path, success); np.save(os.path.join(os.path.dirname(path), 'es_log.npy'), np.stack(self.es_log) if self.es_log else np.zeros((0, self.n, 2)))
