"""Policy process for the real-robot runtime (2026-11-16).

Runs PolicyCore + Runtime in its own process (rdp_venv) and serves one control step per message, so the robot
side (or the simulator, for validation) stays in its own environment. Protocol (pickle, length-prefixed, the same
helpers as the simulation server):
  on connect   -> constants (dataset_obs_steps, obs_temporal_downsample_ratio, original_horizon, obs_keys, ...)
  {'cmd': 'begin', 'frame': f}                    start an episode with observation f
  {'cmd': 'act', 'frame': f, 'success_prev': s}   one control step -> {'action', 'policy', 'residual', 'plan_id',
                                                  'xf_w', 'act_seconds'}
  {'cmd': 'end', 'log_path': p}                   end the episode, optionally save the runtime log (npz)
  {'cmd': 'close'}
f is a dict of arrays with batch dimension N (wrist [front only for the simulation config], state, contact_8d_right, tactile_force_field_right).
usage: rr_server.py --config cfg.json --port P [--realtime] [--residual]
robot_command_sent: false (this process never talks to a robot).
"""
import argparse
import json
import pathlib
import socket
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from policy_core import PolicyCore, S      # noqa: E402
from runtime import Runtime, SimResidual   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--realtime', action='store_true'); ap.add_argument('--n_envs', type=int, default=1)
    a = ap.parse_args(); cfg = json.load(open(a.config))
    core = PolicyCore(cfg['ldp_ckpt'], cfg['at_ckpt'], cfg['verifier_ckpts'], evid_est=cfg.get('evid_est'), n_envs=a.n_envs,
                      coh=dict(lam=float(cfg.get('coh_lambda', 0.02)), m=4, d=int(cfg.get('delay', 2))),
                      # 11-19: rule from the config, default = current system (blocking only: k 999, fallback 2). The 11-16
                      # validation configs (rrval_20261116) ran the old belief rule k 2; set "evid_k": 2 to reproduce them.
                      evid=(None if not cfg.get('evid_est') else dict(k=int(cfg.get('evid_k', 999)), perm_k=-1, theta=1.0, suppress=True,
                                                                     fallback=int(cfg.get('evid_fallback', 2)))),
                      stall=dict(mode=cfg.get('stall_mode', 'mm'), ratio=float(cfg.get('stall_ratio', 0.148)), mm=0.003),
                      **({'arm': cfg['arm']} if cfg.get('arm') else {}))
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', a.port)); srv.listen(1)
    print(f'[rr_server] listening on 127.0.0.1:{a.port} realtime={a.realtime} clip_sample disabled={cfg.get("no_clip", True)}', file=sys.stderr, flush=True)
    while True:
        conn, _ = srv.accept(); rt = None
        try:
            S.send_msg(conn, dict(core.constants))
            while True:
                req = S.recv_msg(conn); c = req.get('cmd')
                if c == 'close':
                    break
                if c == 'begin':
                    n = int(np.asarray(req['frame']['state']).shape[0])
                    res = SimResidual(cfg['residual_ckpt'], n) if cfg.get('residual_ckpt') else None
                    if rt is not None:
                        rt.close()
                    rt = Runtime(core, residual=res, delay=int(cfg.get('delay', 2)), xfade=int(cfg.get('xfade', 0)), realtime=a.realtime,
                                 policy_seed=req.get('policy_seed'))
                    rt.begin(req['frame']); S.send_msg(conn, {'ok': True})
                elif c == 'act':
                    act = rt.act(req['frame'], req.get('success_prev'))
                    L = rt.log
                    S.send_msg(conn, {'action': act, 'policy': L['cmd_policy'][-1], 'residual': L['cmd_residual'][-1], 'plan_id': L['plan_id'][-1],
                                      'xf_w': L['xf_w'][-1], 'act_seconds': L['act_seconds'][-1]})
                elif c == 'end':
                    if req.get('log_path') and rt is not None:
                        L = rt.log
                        np.savez_compressed(req['log_path'], **{k: np.asarray(L[k]) for k in ('t', 'plan_id', 'xf_w', 'act_seconds', 'swap_step', 'swap_plan_age', 'plan_seconds')})
                    S.send_msg(conn, {'ok': True})
                else:
                    S.send_msg(conn, {'error': f'unknown cmd {c}'})
        except (ConnectionError, EOFError):
            pass
        finally:
            conn.close()


if __name__ == '__main__':
    main()
