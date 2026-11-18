"""Real-time timing test of realrobot/runtime.py without hardware (FakeRobot), on this machine's GPU.

Runs the full policy (generator + Q + belief rule + continuity term, residual optional) in realtime mode at 15 Hz
for a few episodes and reports: per-step compute time (act), planner time, plan age at swap (steps from its
observation to taking effect), loop slack and overruns. Images are random, so the robot does nothing sensible;
only timing is measured. usage: timing_test.py config.json out.json [episodes] [steps]
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/dyros-recruit/mcy_ws/manifeel/manifeel/realrobot')
from policy_core import PolicyCore          # noqa: E402
from robot_iface import FakeRobot           # noqa: E402
from runtime import Runtime, SimResidual    # noqa: E402

cfg = json.load(open(sys.argv[1])); out = sys.argv[2]
n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 3; steps = int(sys.argv[4]) if len(sys.argv) > 4 else 300
core = PolicyCore(cfg['ldp_ckpt'], cfg['at_ckpt'], cfg['verifier_ckpts'], evid_est=cfg.get('evid_est'), n_envs=1,
                  stall=dict(mode=cfg.get('stall_mode', 'mm'), ratio=float(cfg.get('stall_ratio', 0.148)), mm=0.003))
res = SimResidual(cfg['residual_ckpt'], 1) if cfg.get('residual_ckpt') else None
R = {}
for xf in (0, int(cfg.get('xfade', 3))):
    acts, plans, ages, slack = [], [], [], []
    for ep in range(n_ep):
        rt = Runtime(core, residual=res, delay=int(cfg.get('delay', 2)), xfade=xf, realtime=True, policy_seed=1000 + ep)
        log = rt.run_episode(FakeRobot(seed=ep), steps)
        rt.close()
        acts += log['act_seconds'][5:]; plans += log['plan_seconds'][1:]; ages += log['swap_plan_age'][1:]; slack += log['slack_s'][5:]
    q = lambda x: {p: round(float(np.percentile(x, p)), 4) for p in (50, 90, 99, 100)}
    R[f'xfade{xf}'] = dict(act_seconds=q(acts), plan_seconds=q(plans), plan_age_steps=dict(zip(*np.unique(ages, return_counts=True))) if ages else {},
                           overrun_share=round(float(np.mean(np.array(slack) < 0)), 4), slack_seconds=q(slack))
    R[f'xfade{xf}']['plan_age_steps'] = {int(k): int(v) for k, v in R[f'xfade{xf}']['plan_age_steps'].items()}
json.dump(R, open(out, 'w'), indent=1); print(json.dumps(R))
