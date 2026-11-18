"""Per-refresh latent verifier: shared state encoder with V (value), d (correction direction) and
dueling Q heads over K candidate plans. Used by train_verifier_vq.py and policy_server.py."""
import torch
from torch import nn


class Verifier(nn.Module):
    def __init__(self, dv=2078, dp=21, dt=0, dc=12, h=256, use_align=True, use_vision=True):
        super().__init__()
        self.dt = dt
        self.use_align = use_align
        self.use_vision = use_vision
        self.v_in = nn.Sequential(nn.Linear(dv, h), nn.SiLU(), nn.Dropout(0.1))
        self.s = nn.Sequential(nn.Linear(h + dp + dt + 1, h), nn.SiLU(), nn.Dropout(0.1), nn.Linear(h, h), nn.SiLU())
        self.V = nn.Linear(h, 1)
        self.d = nn.Linear(h, 2)
        self.A = nn.Sequential(nn.Linear(h + dc + (1 if use_align else 0), 128), nn.SiLU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, V, P, T, stuck, cand, rawxy):
        """V (B,dv) P (B,dp) T (B,dt) stuck (B,) cand (B,K,dc) standardized; rawxy (B,K,2) raw planned xy."""
        vin = self.v_in(V if self.use_vision else torch.zeros_like(V))
        h = self.s(torch.cat([vin, P, T, stuck[:, None]], 1))
        v = self.V(h)[:, 0]
        dd = self.d(h)
        du = dd / (dd.norm(dim=-1, keepdim=True) + 1e-6)
        ru = rawxy / (rawxy.norm(dim=-1, keepdim=True) + 1e-9)
        align = (ru * du[:, None]).sum(-1, keepdim=True)
        hk = h[:, None].expand(-1, cand.shape[1], -1)
        parts = [hk, cand] + ([align] if self.use_align else [])
        A = self.A(torch.cat(parts, -1))[..., 0]
        Q = v[:, None] + A - A.mean(1, keepdim=True)
        return v, dd, Q, A
