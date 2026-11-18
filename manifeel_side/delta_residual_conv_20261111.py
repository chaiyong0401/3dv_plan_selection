"""delta_residual_conv_20261111 (new 2026-11-11; link_cl_20261111): hand-frame gate-free residual (DeltaResidualHF) whose network also has the
convergence head of conv_cls_20261111 (two logits 'correction needed' xy / rot on the shared trunk). After every step self.head (n, 2) holds
the logits of the alive envs (dead envs = +50). The correction output path is unchanged. robot_command_sent: false."""
import numpy as np
import torch
from delta_residual_gatefree_20261106 import DeltaResidualGF
from delta_residual_ms_rld_20261024 import ResMLP, LD, IMG, FF, RHO
from delta_residual_handframe_20261106 import HandFrameNet, XY_SLOTS


class ResConv(ResMLP):
    def __init__(self, d):
        super().__init__(d=d); self.head = torch.nn.Linear(256, 2)

    def both(self, x):
        h = self.net[:-1]((x - self.mu) / self.sd); return RHO * torch.tanh(self.net[-1](h)), self.head(h)


class HandFrameConv(HandFrameNet):
    def forward(self, x):
        q = x[:, 3:7]; psi = torch.atan2(2 * (q[:, 0] * q[:, 1] + q[:, 3] * q[:, 2]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        c, s = torch.cos(psi), torch.sin(psi); x = x.clone()
        for i, j in XY_SLOTS:
            vx, vy = x[:, i].clone(), x[:, i + 1].clone()
            x[:, i], x[:, i + 1] = c * vx + s * vy, -s * vx + c * vy
        d, lg = self.net.both(x); self.last_logits = lg.detach(); out = d.clone()
        out[:, 0], out[:, 1] = c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]
        return out


class DeltaResidualHFC(DeltaResidualGF):
    def __init__(self, n, mode, ckpt=None):
        assert mode == 'deploy'
        self.n = n; super().__init__(n, 'collect'); self.mode = 'deploy'
        ck = torch.load(ckpt, map_location=self.dev); assert ck['variant'] == 'HF_CONV', ck['variant']
        net = ResConv(LD + IMG + FF).to(self.dev).eval(); net.load_state_dict(ck['model'])
        self.net = HandFrameConv(net).to(self.dev).eval(); self.head = None

    def step(self, *a, **k):
        self.net.last_logits = None; out = super().step(*a, **k); h = np.full((self.n, 2), 50.0, np.float32)
        if self.net.last_logits is not None:
            idx = np.where(self.alive)[0]; lg = self.net.last_logits.cpu().numpy(); assert len(idx) == len(lg), (len(idx), len(lg)); h[idx] = lg
        self.head = h
        return out
