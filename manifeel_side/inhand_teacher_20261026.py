"""Keypoint-referenced scripted teacher for single-socket TacSL insertion tasks (USB, Power, PegReorientation).

New file (2026-10-26). Purpose: Stage 0 of the in-hand rotation study -- how often / how much the plug rotates in
the hand while it is in contact, under realistic first-contact offsets.

Success in these tasks = mean distance of 4 keypoints on a 0.5 m line along the plug local z (centred on plug_pos) vs
the same line on the socket (centred on socket_pos, insertion_frac 1.0) below close_error_thresh.  So the teacher
servos the plug tip (plug_pos) onto the socket origin and the plug z axis onto the socket z axis.

Per env
  HOVER  : tip target = goal + z * clear_h (travel at that height), axis servo; INSERT when lateral error < align_tol
           and axis error < rot_tol for align_steps steps (or hover_timeout).
  INSERT : z target ramps down by descend_rate per step to goal - floor_below; xy target = goal + offset, where the
           offset (sampled per env at reset, probability p_offset, magnitude U[off_min, off_max], random direction)
           is held until the plug-socket contact force first exceeds contact_n, then for catch_steps more steps,
           then removed (the servo slides the plug back over the hole).
  ref    : 'live'   -> uses the privileged CURRENT plug pose every step (in-hand changes are compensated);
           'frozen' -> plug pose is predicted from the EE pose and the in-hand pose captured at t=0
                       (an estimate-once-and-commit controller: in-hand changes after t=0 are not seen).

Action convention (= tacsl_task_*._apply_actions_as_ctrl_targets): 6-d per-step deltas in [-1, 1]; position delta
= pos_scale * a[:3] (m), rotation delta = rot_scale * a[3:6] (world axis-angle, rad).  robot_command_sent: false.
"""
from types import SimpleNamespace

import numpy as np
import torch
from isaacgym import torch_utils as tu

HOVER, INSERT = 0, 1

DEFAULT_PARAMS = dict(
    kp_xy=0.6, kp_z=0.35, kp_rot=0.5, pos_scale=0.01, rot_scale=0.05,
    clear_h=0.03,           # m: hover height of the tip above the goal point
    far_xy=0.003,           # m: lateral error above which the tip stays at clear_h
    align_tol=0.0005, rot_tol=0.02, align_steps=2, hover_timeout=120,
    descend_rate=0.0005,    # m/step
    floor_below=0.004,      # m below goal: end of the z ramp
    press_max=0.0015,       # m: z target never more than this below the (controller's) tip -> bounded press
    p_offset=0.0, off_min=0.0015, off_max=0.004,
    contact_n=0.5, catch_steps=10,
    ref='live',
    yaw_align=True,         # also servo the plug x axis onto the socket x axis (180-deg symmetric section)
)


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def plug_socket_force(task):
    return task.contact_force_pairwise[:, task.plug_body_id_env, task.socket_body_id_env]


def inhand_pose(task):
    """Plug pose in the fingertip-midpoint frame: (quat xyzw, pos)."""
    qc = tu.quat_conjugate(task.fingertip_midpoint_quat)
    return tu.quat_mul(qc, task.plug_quat), tu.quat_rotate(qc, task.plug_pos - task.fingertip_midpoint_pos)


def observe(task):
    """Privileged per-step quantities (numpy)."""
    n = task.num_envs
    ez = torch.tensor([0, 0, 1.0], device=task.device).expand(n, 3)
    q_ih, p_ih = inhand_pose(task)
    g_q, g_p = task.socket_quat, task.socket_pos   # insertion_frac 1.0 -> goal = socket origin
    tff = {s: task.obs_dict.get(f'tactile_force_field_{s}') for s in ('left', 'right')}
    ex = torch.tensor([1.0, 0, 0], device=task.device).expand(n, 3)
    out = dict(tip=_np(task.plug_pos), axis=_np(tu.quat_rotate(task.plug_quat, ez)),
               xaxis=_np(tu.quat_rotate(task.plug_quat, ex)),
               goal=_np(g_p), goal_axis=_np(tu.quat_rotate(g_q, ez)), goal_xaxis=_np(tu.quat_rotate(g_q, ex)),
               ee_pos=_np(task.fingertip_midpoint_pos), ee_quat=_np(task.fingertip_midpoint_quat),
               q_ih=_np(q_ih), p_ih=_np(p_ih), finger=_np(task.gripper_dof_pos),
               f_contact=_np(torch.norm(plug_socket_force(task), dim=-1)))
    for s, v in tff.items():
        if v is not None:
            v = _np(v)
            out[f'tac_{s}_normal'] = v[..., 0].reshape(n, -1).sum(-1)
            out[f'tac_{s}_shear'] = v[..., 1:3].reshape(n, -1, 2).sum(1)
    return out


def _rot_err(a, b):
    """World axis-angle rotating unit vector a onto b (N,3)."""
    c = np.cross(a, b)
    s = np.linalg.norm(c, axis=-1, keepdims=True)
    ang = np.arctan2(s, np.sum(a * b, -1, keepdims=True))
    return np.where(s > 1e-9, c / np.maximum(s, 1e-9) * ang, 0.0)


def _qrot(q, v):
    return _np(tu.quat_rotate(torch.as_tensor(q), torch.as_tensor(v)))


class InhandTeacher:
    def __init__(self, num_envs, params=None, seed=0):
        self.p = SimpleNamespace(**{**DEFAULT_PARAMS, **(params or {})})
        self.n = num_envs
        rng = np.random.default_rng(seed)
        has = rng.random(num_envs) < self.p.p_offset
        mag = rng.uniform(self.p.off_min, self.p.off_max, num_envs)
        th = rng.uniform(0, 2 * np.pi, num_envs)
        self.offset = np.stack([np.cos(th), np.sin(th)], -1) * (mag * has)[:, None]
        self.has_offset = has
        self.phase = np.full(num_envs, HOVER)
        self.align_count = np.zeros(num_envs, np.int64)
        self.hover_steps = np.zeros(num_envs, np.int64)
        self.first_contact = np.full(num_envs, -1)
        self.offset_on = np.ones(num_envs, bool)
        self.z_cmd = np.zeros(num_envs, np.float32)
        self.ih0 = None
        self.t = 0

    def _plug_ref(self, o):
        """(tip, axis) used for control: live privileged, or predicted from EE + in-hand pose at t=0."""
        if self.p.ref == 'live':
            return o['tip'], o['axis'], o['xaxis']
        q0, p0 = self.ih0
        tip = o['ee_pos'] + _qrot(o['ee_quat'], p0)
        qp = _np(tu.quat_mul(torch.as_tensor(o['ee_quat']), torch.as_tensor(q0)))
        axis = _qrot(qp, np.tile([0, 0, 1.0], (self.n, 1)).astype(np.float32))
        xaxis = _qrot(qp, np.tile([1.0, 0, 0], (self.n, 1)).astype(np.float32))
        return tip, axis, xaxis

    def act(self, o):
        p, n = self.p, self.n
        if self.ih0 is None:
            self.ih0 = (o['q_ih'].copy(), o['p_ih'].copy())
        tip, axis, xaxis = self._plug_ref(o)
        g, ga = o['goal'], o['goal_axis']
        rot = _rot_err(axis, ga)
        rot_mag = np.linalg.norm(rot, axis=-1)
        yaw = np.zeros(n)
        if p.yaw_align:
            # signed angle about the goal axis from the projected plug x to the nearer of +/- socket x
            px = xaxis - np.sum(xaxis * ga, -1, keepdims=True) * ga
            px /= np.maximum(np.linalg.norm(px, axis=-1, keepdims=True), 1e-9)
            gx = o['goal_xaxis'] * np.where(np.sum(px * o['goal_xaxis'], -1, keepdims=True) >= 0, 1.0, -1.0)
            yaw = np.arctan2(np.sum(np.cross(px, gx) * ga, -1), np.sum(px * gx, -1))
            rot = rot + yaw[:, None] * ga
        rot_mag = np.sqrt(rot_mag ** 2 + yaw ** 2)
        f = o['f_contact']
        for i in range(n):
            if self.first_contact[i] < 0 and f[i] > p.contact_n:
                self.first_contact[i] = self.t
            if self.first_contact[i] >= 0 and self.t - self.first_contact[i] >= p.catch_steps:
                self.offset_on[i] = False
        tgt_xy = g[:, :2] + self.offset * self.offset_on[:, None]
        err_xy = tgt_xy - tip[:, :2]
        lat = np.linalg.norm(g[:, :2] - tip[:, :2], axis=-1)   # alignment test ignores the injected offset
        lat_cmd = np.linalg.norm(err_xy, axis=-1)
        for i in range(n):
            if self.phase[i] == HOVER:
                self.hover_steps[i] += 1
                at_h = abs(tip[i, 2] - (g[i, 2] + p.clear_h)) < 0.002
                ok = lat_cmd[i] < p.align_tol and rot_mag[i] < p.rot_tol and at_h
                self.align_count[i] = self.align_count[i] + 1 if ok else 0
                if self.align_count[i] >= p.align_steps or (self.hover_steps[i] >= p.hover_timeout and lat_cmd[i] < p.far_xy):
                    self.phase[i] = INSERT
                    self.z_cmd[i] = tip[i, 2]
        a = np.zeros((n, 6), np.float32)
        a[:, 0:2] = np.clip(p.kp_xy * err_xy / p.pos_scale, -1, 1)
        a[:, 3:6] = np.clip(p.kp_rot * rot / p.rot_scale, -1, 1)
        z_hover = g[:, 2] + p.clear_h
        z_hover = np.where(lat_cmd > p.far_xy, np.maximum(z_hover, tip[:, 2]), z_hover)
        az_hover = np.clip(p.kp_z * (z_hover - tip[:, 2]) / p.pos_scale, -1, 1)
        ins = self.phase == INSERT
        self.z_cmd = np.where(ins, np.maximum.reduce([self.z_cmd - p.descend_rate, g[:, 2] - p.floor_below,
                                                      tip[:, 2] - p.press_max]), self.z_cmd)
        az_ins = np.clip((self.z_cmd - tip[:, 2]) / p.pos_scale, -1, 1)
        a[:, 2] = np.where(ins, az_ins, az_hover)
        self.t += 1
        info = dict(phase=self.phase.copy(), lat_err=lat.astype(np.float32), rot_err=rot_mag.astype(np.float32),
                    offset_on=self.offset_on.copy())
        return a, info


def run_episode(task, teacher, steps, on_step=None):
    """Roll `steps` control steps from the current (already reset) state. Returns stacked logs (T, N, ...)."""
    task.refresh_all_tensors()
    task.compute_observations()
    logs = {}
    for t in range(steps):
        o = observe(task)
        o['success'] = _np(task._check_success()).astype(bool)
        a, info = teacher.act(o)
        if on_step is not None:
            on_step(t, o, a, info)
        for k, v in {**o, **info, 'action': a}.items():
            logs.setdefault(k, []).append(v)
        task.step(torch.as_tensor(a, device=task.device))
        task.refresh_all_tensors()
    o = observe(task)
    o['success'] = _np(task._check_success()).astype(bool)
    out = {k: np.stack(v, 0) for k, v in logs.items()}
    out['success_final'] = o['success']
    return out
