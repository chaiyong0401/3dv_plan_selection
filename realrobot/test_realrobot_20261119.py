"""CPU tests for the 2026-11-19 realrobot changes (no GPU, no checkpoints, no robot). python test_realrobot_20261119.py"""
import os, sys, tempfile, types
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime import RealResidual, Runtime
from robot_iface import FakeRobot
from recorder import EpisodeRecorder, build_zarr
import torch

def q_yaw(a):   # x y z w quaternion of a pure yaw
    return np.array([[0, 0, np.sin(a / 2), np.cos(a / 2)]], np.float32)

# 1. ER3 sign, clipping, z asymmetry, hand -> base rotation, stop after success
r = RealResidual(net=None, feat=None, n=1, errnet=object(), er3_axes=('rz',), rho=0.2, z_down=0.05, z_up=0.2)
r.reset(None)
st = np.concatenate([np.zeros((1, 3), np.float32), q_yaw(np.pi / 2)], 1)
out = r.post(np.array([[0.1, 0.0, -0.3, 0.5, -0.5, -0.07]]), {'rz': np.array([3.0])}, st)
assert np.allclose(out[0, 5], -0.07), out            # |rz| with sign(-e) = -1
assert np.isclose(out[0, 2], -0.05) and np.isclose(out[0, 3], 0.2) and np.isclose(out[0, 4], -0.2), out
assert np.allclose(out[0, :2], [0.0, 0.1], atol=1e-6), out   # hand x -> base y at psi = 90 deg
out = r.post(np.array([[0, 0, 0.5, 0, 0, 0.07]]), {'rz': np.array([-2.0])}, st)
assert np.isclose(out[0, 2], 0.2) and np.isclose(out[0, 5], 0.07), out
r.note_success([True]); assert np.all(r.post(np.ones((1, 6)), {'rz': np.ones(1)}, st) == 0)
r2 = RealResidual(net=None, feat=None, n=1)          # no errnet -> no ER3
r2.reset(None); assert np.isclose(r2.post(np.array([[0, 0, 0, 0, 0, 0.1]]), {}, st)[0, 5], 0.1)

# 2. correction() end to end with tiny stand-in networks
net = torch.nn.Linear(4, 6); errnet = lambda x: {'rz': x[:, 0] * 0 + 5.0}
r3 = RealResidual(net=net, feat=lambda a, f, z: np.ones((1, 4), np.float32), n=1, errnet=errnet)
r3.reset(None); c = r3.correction(np.zeros((1, 6)), {'state': st}, None); assert c.shape == (1, 6) and c[0, 5] <= 0

# 3. FakeRobot wrist-only, Runtime loop with a stub core (no generator), recorder without 'front'
fr = FakeRobot(seed=0); fr.start_episode(); f0 = fr.read(); assert 'front' not in f0 and 'wrist' in f0
class StubCore:
    cfg = {'latent_refresh': 12}; n = 1; feats_now = None
    def reset(self, n_envs=None): pass
    def push_frame(self, f, fill=False): pass
    def snapshot(self): return {}
    def plan(self, snap, seed=None): return torch.zeros(1), {'chosen': np.zeros(1, int)}
    def decode(self, latent, k): return torch.full((1, 6), 0.01)
    def note_action(self, a, s): pass
    def step_features(self): self.feats_now = np.zeros((1, 4), np.float32)
rt = Runtime(StubCore(), residual=r3, delay=2, realtime=False)
log = rt.run_episode(FakeRobot(seed=1), 30, hz=1000.0)
assert len(log['cmd_sent']) == 30 and log['swap_plan_age'][1:] and all(a == 2 for a in log['swap_plan_age'][1:]), log['swap_plan_age']
d = tempfile.mkdtemp(); rec = EpisodeRecorder(d); rec.start({'arm': 'test'})
for _ in range(3): rec.add(f0, np.zeros(6), 'policy')
p = rec.finish(False); z = np.load(p); assert 'front' not in z.files and z['wrist'].shape == (3, 256, 256, 3)
build_zarr([p], d)

# 4. arm table of policy_core without importing torch-heavy server code: parse the ARMS literal
import ast
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'policy_core.py')).read()
arms = eval(src.split('ARMS = ', 1)[1].split('\n}\n', 1)[0] + '\n}', {'__builtins__': {}, 'dict': dict})
assert set(arms) == {'A0_generator', 'A1_generator_residual', 'A2_full', 'A3_no_corrector', 'A4_no_rule', 'A5_no_q', 'A6_full_er3'}
assert arms['A3_no_corrector']['residual'] is False and arms['A4_no_rule']['rule'] is False and arms['A5_no_q']['q'] == 'zero'
assert "evid=dict(k=999" in src and "stall=dict(mode='mm'" in src
print('ALL OK')
