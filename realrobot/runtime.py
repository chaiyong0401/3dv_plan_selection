"""Real-robot runtime (2026-11-16): timing around PolicyCore.

Control loop at `hz` (15 Hz). Every `latent_refresh` (12) steps a plan is requested from the observation of that
step. A plan takes effect `delay` (2) steps after its observation, or later if the planner is slower; it is then
decoded with the window that matches "steps since its observation" (latency matching: it enters at the step it
would have reached). Optional (default off since 2026-11-17, xfade_20261116 / rrval: no gain): for `xfade` steps after a swap
the command is (1-w) * old plan + w * new plan, with w = (j+1)/(xfade+1). Smoothing belongs to the controller (rate limit + spline). The residual correction is added after the blend. The loop never waits for the planner.

Interface: begin(frame0) at episode start, then cmd = act(frame_t, success_prev) once per control step.
run_episode() wraps this for a robot object; a simulator bridge can call act() inside its own loop.

Two modes, same code path:
  realtime=False  lockstep: the plan is computed synchronously and swapped exactly `delay` steps later
                  (= the simulation's delayed configuration); used to validate against simulation results.
  realtime=True   the planner runs in a background thread on its own CUDA stream; act() never blocks on it.
                  If the planner is late the current plan keeps running; with no plan at all the command is zero.

Every step is logged (timing, which plan, blend weight, policy / residual / sent command).
Whether anything reaches a robot is decided by the robot interface, not here (robot_command_sent: false).
"""
import collections
import concurrent.futures as cf
import time

import numpy as np
import torch


class Plan:
    __slots__ = ('latent', 't0', 'info', 'id', 'ready_s')

    def __init__(self, latent, t0, info, pid, ready_s):
        self.latent, self.t0, self.info, self.id, self.ready_s = latent, t0, info, pid, ready_s


class Runtime:
    def __init__(self, core, residual=None, delay=2, xfade=0, realtime=False, policy_seed=None, max_cmd=1.0, log=None):
        self.core, self.residual = core, residual
        self.R = int(core.cfg['latent_refresh']); self.delay = int(delay); self.xfade = int(xfade)
        self.realtime, self.policy_seed, self.max_cmd = bool(realtime), policy_seed, float(max_cmd)
        self.log = log if log is not None else collections.defaultdict(list)
        self.pool = cf.ThreadPoolExecutor(max_workers=1) if self.realtime else None
        self.stream = torch.cuda.Stream() if (self.realtime and torch.cuda.is_available()) else None

    # ------------------------------------------------------------------ planner
    def _plan_job(self, snap, t0, pid):
        s0 = time.perf_counter(); seed = None if self.policy_seed is None else self.policy_seed + t0
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                lat, info = self.core.plan(snap, seed=seed)
            self.stream.synchronize()
        else:
            lat, info = self.core.plan(snap, seed=seed)
        return Plan(lat, t0, info, pid, time.perf_counter() - s0)

    # ------------------------------------------------------------------ episode
    def begin(self, frame0):
        self.core.reset(n_envs=np.asarray(frame0['state']).shape[0]); self.core.push_frame(frame0, fill=True)
        self.t = 0; self.cur = self.old = self.pending = None; self.swap_t = -10 ** 9; self.pid = 0
        if self.residual is not None:
            self.residual.reset(frame0)

    def act(self, frame, success_prev=None):
        """frame: observation of this step (pushed here except at t = 0, which begin() already pushed).
        success_prev: success flags after the previous step (stops the residual, as in simulation)."""
        core, t, L = self.core, self.t, self.log
        tick0 = time.perf_counter()
        if t > 0:
            core.push_frame(frame)
        if success_prev is not None and self.residual is not None:
            self.residual.note_success(success_prev)
        # ---- refresh request
        if t % self.R == 0:
            snap = core.snapshot()
            self.pending = (t, self.pool.submit(self._plan_job, snap, t, self.pid) if self.realtime else self._plan_job(snap, t, self.pid))
            self.pid += 1
        # ---- swap when the new plan is ready and old enough (first plan: at once)
        if self.pending is not None:
            t0, p = self.pending
            ready = (not self.realtime) or p.done()
            if ready and (self.cur is None or t - t0 >= self.delay):
                newp = p.result() if self.realtime else p
                self.old, self.cur, self.swap_t, self.pending = self.cur, newp, t, None
                L['swap_step'].append(t); L['swap_plan_age'].append(t - t0); L['plan_seconds'].append(newp.ready_s)
                L['plan_chosen'].append(newp.info.get('chosen')); L['plan_info'].append(newp.info)
        # ---- policy command (blended across a swap)
        w = np.nan
        if self.cur is None:
            a_pol = np.zeros((core.n, 6), np.float32)
        else:
            a_new = core.decode(self.cur.latent, t - self.cur.t0)
            j = t - self.swap_t
            if self.old is not None and j < self.xfade:
                w = (j + 1) / (self.xfade + 1)
                a_old = core.decode(self.old.latent, t - self.old.t0)
                a_t = (1.0 - w) * a_old + w * a_new
                L['xf_step'].append(t); L['xf_old'].append(a_old.float().cpu().numpy()); L['xf_new'].append(a_new.float().cpu().numpy())
            else:
                a_t = a_new
            a_pol = a_t.float().cpu().numpy()
            core.note_action(a_t, frame['state'])
        # ---- residual (after the blend)
        a_res = np.zeros_like(a_pol)
        if self.residual is not None and self.cur is not None:
            core.step_features()
            a_res = self.residual.correction(a_pol, frame, core.feats_now)
        cmd = np.clip(a_pol + a_res, -self.max_cmd, self.max_cmd).astype(np.float32)
        L['t'].append(t); L['plan_id'].append(-1 if self.cur is None else self.cur.id); L['xf_w'].append(w)
        L['cmd_policy'].append(a_pol); L['cmd_residual'].append(a_res); L['cmd_sent'].append(cmd)
        L['act_seconds'].append(time.perf_counter() - tick0)
        self.t += 1
        return cmd

    def run_episode(self, robot, max_steps, hz=15.0):
        """Real-time loop for a robot object with start_episode / read / send / success / end_episode."""
        dt = 1.0 / float(hz)
        robot.start_episode(); frame = robot.read(); self.begin(frame)
        ok_prev = None; t_next = time.perf_counter()
        for _ in range(max_steps):
            cmd = self.act(frame, ok_prev)
            robot.send(cmd)
            t_next += dt; sl = t_next - time.perf_counter(); self.log['slack_s'].append(sl)
            if sl > 0:
                time.sleep(sl)
            else:
                t_next = time.perf_counter()          # overrun: no catch-up burst
            frame = robot.read(); ok_prev = robot.success()
            if np.all(ok_prev):
                break
        robot.end_episode()
        return self.log

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)


class SimResidual:
    """Adapter for the simulation-trained USB residual (DeltaResidualHFC, wrist force blinded). The real XELA 4x4
    sensor needs a residual trained on that input; this adapter serves the validation against simulation."""

    def __init__(self, ckpt, n):
        import sys
        p = '/home/dyros-recruit/mcy_ws/manifeel/manifeel'
        if p not in sys.path:
            sys.path.insert(0, p)
        from delta_residual_conv_20261111 import DeltaResidualHFC
        self.n, self.ckpt = n, ckpt; self.res = None

        class _Src:
            last_feats = None
        self._Src = _Src; self._cls = DeltaResidualHFC

    def reset(self, frame0):
        self.res = self._cls(self.n, 'deploy', self.ckpt); self.src = self._Src(); self.res.feats_src = self.src
        self.wrench, self.success = [], []
        self.ff_init = np.asarray(frame0['tactile_force_field_right'], np.float32).copy()

    def note_success(self, ok):
        self.success.append(np.asarray(ok))

    def correction(self, a_pol, frame, feats):
        n = self.n; self.src.last_feats = feats
        z3, q = np.zeros((n, 3), np.float32), np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1))
        out = self.res.step(a_pol, {'state': frame['state']}, frame['contact_8d_right'], self.wrench, self.success,
                            z3, z3[:, None], frame['tactile_force_field_right'], self.ff_init, q, q[:, None])
        self.wrench.append(np.zeros((n, 6), np.float32))     # wrist force blinded in this residual; history kept for its window
        return np.asarray(out, np.float32) - np.asarray(a_pol, np.float32)


class RealResidual:
    """Corrector for the real robot (2026-11-19, plan: qsel_core/REAL_ROBOT_MS_PLAN.md). Not trained yet.

    net(x) -> (N, 6) correction in the HAND frame (x, y rotated by the hand heading psi; z, rx, ry, rz unchanged),
    trained on human take-over corrections (human command - generator command), all 6 axes incl. z.
    feat(a_pol, frame, feats) -> x builds the net input; it must be the same function the training used.

    Output, in this order:
      1. ER3 (optional, errnet given): for each axis in er3_axes the magnitude stays the residual's and the sign comes
         from the absolute-error head: out[a] = |out[a]| * sign(-e_hat[a]). errnet(x) -> dict axis -> (N,) signed error
         (hand frame, e.g. 'rz' in degrees folded to the nearer 180, 'x' / 'y' in mm). Simulation (USB): rz only,
         +8.3 pp over 3 seeds (errsim3x_20261118). x / y sign replacement is UNTESTED, hence er3_axes=('rz',).
      2. clip: x, y, rx, ry, rz to +-rho (0.2, simulation); z asymmetric to [-z_down, +z_up] (down = negative z, as the
         generator's descending command). z_down / z_up are safety bounds to set with the impedance stiffness.
      3. hand frame -> base frame for x, y.
    Stops after success (as in simulation). robot_command_sent: false.
    """
    AX = {'x': 0, 'y': 1, 'z': 2, 'rx': 3, 'ry': 4, 'rz': 5}

    def __init__(self, net, feat, n=1, errnet=None, er3_axes=('rz',), rho=0.2, z_down=0.05, z_up=0.2, device='cpu'):
        self.net, self.feat, self.n, self.errnet = net, feat, int(n), errnet
        self.er3_axes = tuple(er3_axes) if errnet is not None else ()
        self.rho, self.z_down, self.z_up, self.device = float(rho), float(z_down), float(z_up), device
        self.log = collections.defaultdict(list)

    @staticmethod
    def psi(state):
        q = np.asarray(state, np.float64)[:, 3:7]          # x y z w (same as delta_residual_handframe_20261106)
        return np.arctan2(2 * (q[:, 0] * q[:, 1] + q[:, 3] * q[:, 2]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))

    def reset(self, frame0):
        self.alive = np.ones(self.n, bool)

    def note_success(self, ok):
        self.alive &= ~np.asarray(ok, bool).reshape(self.n)

    def post(self, out_hand, e_hat, state):
        """Steps 1-3 on numpy arrays; separated so it can be tested without a network."""
        out = np.array(out_hand, np.float32, copy=True)
        for a in self.er3_axes:
            i = self.AX[a]; out[:, i] = np.abs(out[:, i]) * np.sign(-np.asarray(e_hat[a], np.float32))
        for i in (0, 1, 3, 4, 5):
            out[:, i] = np.clip(out[:, i], -self.rho, self.rho)
        out[:, 2] = np.clip(out[:, 2], -self.z_down, self.z_up)
        p = self.psi(state); c, s = np.cos(p), np.sin(p); x, y = out[:, 0].copy(), out[:, 1].copy()
        out[:, 0], out[:, 1] = c * x - s * y, s * x + c * y
        out[~self.alive] = 0.0
        return out

    @torch.no_grad()
    def correction(self, a_pol, frame, feats):
        x = torch.as_tensor(self.feat(a_pol, frame, feats), dtype=torch.float32, device=self.device)
        out = self.net(x).float().cpu().numpy()
        e_hat = {k: v.float().cpu().numpy() for k, v in self.errnet(x).items()} if self.errnet is not None else {}
        res = self.post(out, e_hat, frame['state'])
        self.log['res_hand'].append(out); self.log['res_base'].append(res)
        for k, v in e_hat.items():
            self.log[f'e_hat_{k}'].append(v)
        return res
