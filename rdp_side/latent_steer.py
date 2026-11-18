"""Learned bounded steering on top of a FIXED LDP + AT + learned_q selector (latent_steer_20261012).

At every learned_q refresh (verifier active, step >= 36) the server picks z0 among K=8 draws exactly as before, then
  mode 'latent': z = z0 + scale * (a * A * sig) @ U      (U: m x D within-state PCA basis of LDP draws, sig: PC std)
  mode 'action': r = scale * a * b, added to every executed action of the 12-step window (after the reactive AT decode)
with a in [-1, 1]^m. a comes from uniform exploration (collection) or a learned actor tanh(f(s)) (evaluation).
The AT keeps re-decoding the held latent with the newest tactile frame every step; the refresh interval is unchanged.
scale 0 reproduces the unsteered execution bitwise (z0 + 0, action + 0); the server counts that identity per refresh.
"""
import numpy as np
import torch
from torch import nn


class SteerNet(nn.Module):
    """Shared shape for the actor (out = m, tanh-bounded at use) and the value baseline (out = 1)."""

    def __init__(self, dv=2078, dp=21, dt=26, dz=128, out=16, h=256):
        super().__init__()
        self.v_in = nn.Sequential(nn.Linear(dv, h), nn.SiLU(), nn.Dropout(0.1))
        self.body = nn.Sequential(nn.Linear(h + dp + dt + dz, h), nn.SiLU(), nn.Dropout(0.1),
                                  nn.Linear(h, h), nn.SiLU(), nn.Linear(h, out))

    def forward(self, V, P, T, Z):
        return self.body(torch.cat([self.v_in(V), P, T, Z], 1))


def load_steer(args_mode, source, scale, basis_path, actor_ckpt, seed, device, action_dim):
    bz = np.load(basis_path)
    st = {'mode': args_mode, 'source': source, 'scale': float(scale), 'seed': int(seed), 'n_identity_checked': 0,
          'n_identity_failed': 0}
    if args_mode == 'latent':
        st['U'] = torch.as_tensor(bz['U'], dtype=torch.float32, device=device)            # (m, D)
        st['amp'] = torch.as_tensor(bz['A'] * bz['sig'], dtype=torch.float32, device=device)  # (m,)
        st['m'] = int(bz['U'].shape[0])
    else:
        b = np.asarray(bz['b_action'], np.float32)
        if b.shape != (action_dim,):
            raise ValueError(f'b_action shape {b.shape} != ({action_dim},)')
        st['amp'] = torch.as_tensor(b, device=device)
        st['m'] = action_dim
    if source == 'actor':
        ck = torch.load(actor_ckpt, map_location=device, weights_only=False)
        if ck['mode'] != args_mode or ck['m'] != st['m']:
            raise ValueError('actor checkpoint mode/dim does not match --steer-mode / basis')
        net = SteerNet(dv=ck['dv'], dp=ck['dp'], dt=ck['dt'], dz=ck['dz'], out=ck['m']).to(device)
        net.load_state_dict(ck['state_dict']); net.eval()
        st['actor'] = net
        st['stats'] = {k: (torch.as_tensor(a, device=device, dtype=torch.float32),
                           torch.as_tensor(c, device=device, dtype=torch.float32)) for k, (a, c) in ck['stats'].items()}
    elif source != 'explore':
        raise ValueError(source)
    return st


def steer_coeffs(st, feats, P, T, z0, policy_seed, step):
    """a in [-1, 1]^(N, m). Exploration uses its own numpy generator (torch RNG untouched)."""
    n = z0.shape[0]
    if st['source'] == 'explore':
        rng = np.random.default_rng([st['seed'], int(policy_seed), int(step)])
        return torch.as_tensor(rng.uniform(-1.0, 1.0, (n, st['m'])), dtype=torch.float32, device=z0.device)
    s = st['stats']
    zs = lambda x, k: (x.float() - s[k][0]) / s[k][1]
    with torch.no_grad():
        return torch.tanh(st['actor'](zs(feats, 'V'), zs(P, 'P'), zs(T, 'T'), zs(z0, 'Z')))


def apply_steer(st, a, z0):
    """Returns (z, act_offset or None). Identity at scale 0 is checked bitwise and counted."""
    if st['mode'] == 'latent':
        delta = (st['scale'] * a * st['amp'][None]) @ st['U']
        z = z0 + delta.to(z0.dtype)
        if st['scale'] == 0.0:
            st['n_identity_checked'] += 1
            st['n_identity_failed'] += int(not torch.equal(z, z0))
        return z, None
    r = (st['scale'] * a * st['amp'][None])
    if st['scale'] == 0.0:
        st['n_identity_checked'] += 1
        st['n_identity_failed'] += int(bool((r != 0).any()))
    return z0, r
