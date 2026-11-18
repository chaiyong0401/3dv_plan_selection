"""delta_residual_b2_20261117: per env, the deployed network residual (HF_CONV) until the env has chosen a switch candidate (server reply evid_done,
read from feats_src.last_done), and the combined residual of linres_20261117 (|net| * sign(linear), linres.npz via env RDP_LINRES) after it.
No task variable: the switch flag is the server's own record of its choice (candidate distance > theta). robot_command_sent: false."""
import os
import numpy as np
import torch
from delta_residual_linsign_20261117 import LinSignHF, DeltaResidualLS


class LinSignB2(LinSignHF):
    def forward(self, x):
        q = x[:, 3:7]; psi = torch.atan2(2 * (q[:, 0] * q[:, 1] + q[:, 3] * q[:, 2]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        c, s = torch.cos(psi), torch.sin(psi)
        out_lin = super().forward(x)
        d = self._d_last                                               # hand-frame network output from the same call
        out_net = d.clone(); out_net[:, 0], out_net[:, 1] = c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]
        m = self.mask if getattr(self, 'mask', None) is not None else torch.zeros(len(x), dtype=torch.bool, device=x.device)
        assert len(m) == len(x), (len(m), len(x))
        return torch.where(m[:, None], out_lin, out_net)


def _patch_both(hfc, holder):
    orig = hfc.net.both
    def both(xh):
        d, lg = orig(xh); holder._d_last = d; return d, lg
    hfc.net.both = both


class DeltaResidualB2(DeltaResidualLS):
    def __init__(self, n, mode, ckpt=None):
        super().__init__(n, mode, ckpt)
        ls = self.net; b2 = LinSignB2.__new__(LinSignB2); b2.__dict__ = ls.__dict__; self.net = b2
        _patch_both(b2.hfc, b2); self.b2_log = []

    def step(self, *a, **k):
        alive = self.alive.copy(); sh = a[4] if len(a) > 4 else k.get('success_hist')
        if sh:
            alive &= ~(np.asarray(sh[-1]) > 0)
        idx = np.where(alive)[0]
        dn = getattr(self.feats_src, 'last_done', None)
        done = np.zeros(self.n, bool) if dn is None else np.asarray(dn).astype(bool).reshape(-1)
        self.net.mask = torch.as_tensor(done[idx], device=self.dev)
        self.b2_log.append(done.copy())
        return super().step(*a, **k)

    def save(self, path, succ):
        super().save(path, succ); np.save(str(path).replace('delta_log.npz', 'b2_done.npy'), np.array(self.b2_log, dtype=bool))
