"""delta_residual_linsign_20261117 (linres_20261117 AMEND1): deployed network residual (HF_CONV, residual_conv.pt) whose per-axis SIGN is replaced by the
sign of a ridge model (linres.npz, env RDP_LINRES) on the same hand-frame inputs: out_i = |net_i| * sign(lin_i) (net_i where lin_i == 0). Magnitude and
'when' stay the network's. Convergence-head logits are passed through unchanged. robot_command_sent: false."""
import os
import numpy as np
import torch
from delta_residual_conv_20261111 import DeltaResidualHFC, HandFrameConv
from delta_residual_handframe_20261106 import XY_SLOTS


class LinSignHF(torch.nn.Module):
    def __init__(self, hfc, lin):
        super().__init__(); self.hfc = hfc; self.last_logits = None
        keep = torch.as_tensor(lin['keep_ld'], dtype=torch.long); self.register_buffer('keep', keep)
        for k in ('W', 'b', 'mu', 'sd'):
            self.register_buffer(k, torch.as_tensor(lin[k], dtype=torch.float32))

    def forward(self, x):
        q = x[:, 3:7]; psi = torch.atan2(2 * (q[:, 0] * q[:, 1] + q[:, 3] * q[:, 2]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        c, s = torch.cos(psi), torch.sin(psi); xh = x.clone()
        for i, j in XY_SLOTS:
            vx, vy = xh[:, i].clone(), xh[:, i + 1].clone()
            xh[:, i], xh[:, i + 1] = c * vx + s * vy, -s * vx + c * vy
        d, lg = self.hfc.net.both(xh); self.last_logits = lg.detach()
        f = torch.cat([xh[:, :100][:, self.keep], xh[:, 100:]], 1)
        lin = ((f - self.mu) / self.sd) @ self.W + self.b
        sg = torch.sign(lin); dh = torch.where(sg == 0, d, d.abs() * sg)
        out = dh.clone(); out[:, 0], out[:, 1] = c * dh[:, 0] - s * dh[:, 1], s * dh[:, 0] + c * dh[:, 1]
        return out


class DeltaResidualLS(DeltaResidualHFC):
    def __init__(self, n, mode, ckpt=None):
        super().__init__(n, mode, ckpt)
        lin = np.load(os.environ['RDP_LINRES']); self.net = LinSignHF(self.net, lin).to(self.dev).eval()
