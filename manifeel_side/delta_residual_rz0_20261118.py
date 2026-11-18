"""rz0 (new file, 2026-11-18; rzoff_20261118): DeltaResidualHFC (delta_residual_conv_20261111, unchanged; the Q400B residual) with the
network's rz output (action dim 5) set to 0 at every step. x, y, z, rx, ry outputs, inputs, logging and the convergence head are unchanged.
Purpose: causal test of whether the residual's benefit comes from rz (resfail_20261118 suggested it does not). robot_command_sent: false."""
import torch
from delta_residual_conv_20261111 import DeltaResidualHFC, HandFrameConv


class HandFrameConvRZ0(HandFrameConv):
    def forward(self, x):
        out = HandFrameConv.forward(self, x).clone(); out[:, 5] = 0.0
        return out


class DeltaResidualRZ0(DeltaResidualHFC):
    def __init__(self, n, mode, ckpt=None):
        super().__init__(n, mode, ckpt); self.net.__class__ = HandFrameConvRZ0


if __name__ == '__main__':   # unit check: identical to HandFrameConv except dim 5 == 0
    from delta_residual_ms_rld_20261024 import LD, IMG, FF
    from delta_residual_conv_20261111 import ResConv
    torch.manual_seed(0); net = ResConv(LD + IMG + FF).eval(); a = HandFrameConv(net).eval(); b = HandFrameConv(net).eval(); b.__class__ = HandFrameConvRZ0
    x = torch.randn(7, LD + IMG + FF); q = torch.randn(7, 4); x[:, 3:7] = q / q.norm(dim=1, keepdim=True)
    with torch.no_grad(): ya, yb = a(x), b(x)
    assert torch.allclose(ya[:, :5], yb[:, :5]) and (yb[:, 5] == 0).all() and (ya[:, 5] != 0).any()
    print('rz0 selftest OK')
