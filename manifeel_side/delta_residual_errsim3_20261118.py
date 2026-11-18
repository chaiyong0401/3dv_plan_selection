"""errsim3 (new file, 2026-11-18; errsim3_20261118): DeltaResidualHFC (delta_residual_conv_20261111, unchanged; residual_conv.pt) with ONLY the
SIGN of the rz output replaced: rz = |residual rz| * sign(-e_hat). Magnitude (when / how much) stays the residual's own; direction comes from
the yaw-error estimator errnet (errsim2_20261118/errnet2.pt 'err', env RDP_ERRNET2). No gate, no threshold. e_hat per step -> es_log.npy.
robot_command_sent: false."""
import os, sys, numpy as np, torch
from delta_residual_conv_20261111 import DeltaResidualHFC, HandFrameConv
from delta_residual_errsim_20261118 import to_hand_t
sys.path.insert(0, '/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs/errsim2_20261118')
from train_err2 import OneNet


class HandFrameConvES3(HandFrameConv):
    def forward(self, x):
        out = HandFrameConv.forward(self, x).clone(); eh = self.errnet(to_hand_t(x))
        out[:, 5] = out[:, 5].abs() * torch.sign(-eh); self.es_last = torch.stack([eh, out[:, 5]], 1).detach()
        return out


class DeltaResidualES3(DeltaResidualHFC):
    def __init__(self, n, mode, ckpt=None):
        super().__init__(n, mode, ckpt); ck = torch.load(os.environ['RDP_ERRNET2'], map_location=self.dev); d = ck['d']
        en = OneNet(d, np.zeros(d, np.float32), np.ones(d, np.float32), 'err').to(self.dev).eval(); en.load_state_dict(ck['err'])
        self.net.__class__ = HandFrameConvES3; self.net.errnet = en; self.es_log = []

    def step(self, *a, **k):
        self.net.es_last = None; out = super().step(*a, **k); v = np.full((self.n, 2), np.nan, np.float32)
        if self.net.es_last is not None:
            idx = np.where(self.alive)[0]; v[idx] = self.net.es_last.cpu().numpy()
        self.es_log.append(v); return out

    def save(self, path, success):
        super().save(path, success); np.save(os.path.join(os.path.dirname(path), 'es_log.npy'), np.stack(self.es_log) if self.es_log else np.zeros((0, self.n, 2)))
