"""handframe (new file, 2026-11-06; usb_keyed_s12u_handframe_res_20261106): DeltaResidualGF (delta_residual_gatefree_20261106, unchanged)
with the network wrapped in the hand frame. psi = heading of the EE x-axis in world from the state quaternion (state[3:7], x y z w).
World-frame xy inputs (EE motion over 1 / 12 steps ld[7:9], ld[10:12]; base action ld[93:95]) are rotated by R(-psi) before the
network; the network's xy output is rotated back by R(psi). rz and all other inputs unchanged. Collect mode is not provided (the
s12r collection is reused; to_hand() is the training-side transform). robot_command_sent: false."""
import numpy as np
import torch
import torch.nn as nn
from delta_residual_gatefree_20261106 import DeltaResidualGF
from delta_residual_ms_rld_20261024 import ResMLP, LD, IMG, FF

XY_SLOTS = ((7, 9), (10, 12), (93, 95))


def psi_from_quat(q):
    """q (..., 4) x y z w -> heading of the body x-axis in world (rad); same formula as the xy_head analyses."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))


def rot2(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def to_hand(ld, label=None):
    """numpy: rotate the world-frame xy slots of ld (N, LD) and the label xy (N, 6) into the hand frame."""
    ld = ld.copy(); a = -psi_from_quat(ld[:, 3:7])
    for i, j in XY_SLOTS:
        ld[:, i:j] = rot2(ld[:, i:j], a)
    if label is None:
        return ld
    lb = label.copy(); lb[:, :2] = rot2(lb[:, :2], a)
    return ld, lb


class HandFrameNet(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, x):
        q = x[:, 3:7]; psi = torch.atan2(2 * (q[:, 0] * q[:, 1] + q[:, 3] * q[:, 2]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        c, s = torch.cos(psi), torch.sin(psi); x = x.clone()
        for i, j in XY_SLOTS:
            vx, vy = x[:, i].clone(), x[:, i + 1].clone()
            x[:, i], x[:, i + 1] = c * vx + s * vy, -s * vx + c * vy           # R(-psi)
        d = self.net(x); out = d.clone()
        out[:, 0], out[:, 1] = c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]   # R(psi)
        return out


class DeltaResidualHF(DeltaResidualGF):
    def __init__(self, n, mode, ckpt=None):
        assert mode == 'deploy', 'hand-frame residual: deploy only (collection reused from s12r)'
        self.n = n; super().__init__(n, 'collect')          # base bookkeeping without loading a GF ckpt
        self.mode = 'deploy'
        ck = torch.load(ckpt, map_location=self.dev); assert ck['variant'] == 'HF', ck['variant']
        net = ResMLP(d=LD + IMG + FF).to(self.dev).eval(); net.load_state_dict(ck['model'])
        self.net = HandFrameNet(net).to(self.dev).eval()


if __name__ == '__main__':   # unit checks: numpy and torch transforms agree; a 180 deg hand flips the world output sign
    rng = np.random.default_rng(0); N = 5
    ld = rng.normal(size=(N, LD)).astype(np.float32)
    q180 = np.array([0.9978, -0.0481, 0.0370, -0.0268], np.float32)        # EE pointing down (sample state)
    ld[:, 3:7] = q180
    lh, lb = to_hand(ld, np.tile(np.array([[1, 0, 0, 0, 0, 0.1]], np.float32), (N, 1)))
    class Id(nn.Module):
        def forward(self, x): return torch.cat([x[:, 93:95], torch.zeros(len(x), 4)], 1)   # echo the hand-frame base action xy
    m = HandFrameNet(Id()); x = torch.from_numpy(np.concatenate([ld, np.zeros((N, IMG + FF), np.float32)], 1))
    y = m(x).numpy()
    assert np.allclose(y[:, :2], ld[:, 93:95], atol=1e-5), 'R(psi) R(-psi) must be identity'
    assert np.allclose(lh[:, 93:95], rot2(ld[:, 93:95], -psi_from_quat(ld[:, 3:7])), atol=1e-6)
    zq = np.array([0, 0, 1, 0], np.float32)   # 180 deg about z composed: x-axis heading flips by pi
    def qmul(a, b):
        ax, ay, az, aw = a; bx, by, bz, bw = b
        return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], np.float32)
    qr = qmul(zq, q180); d = (psi_from_quat(qr) - psi_from_quat(q180) + np.pi) % (2 * np.pi) - np.pi
    assert abs(abs(d) - np.pi) < 1e-3, d
    print('handframe selftest OK')
