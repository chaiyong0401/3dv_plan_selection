"""Intervention DAgger (HG-DAgger style) for the manifeel PiH bridge (dagger_r1_20261010, PLAN.md).

The policy drives; a privileged scripted expert takes over only when a failure condition fires, corrects, and hands
back. Only the expert segments are recorded as training data.

  TILT  : plug tilt > tilt_deg AND plug-socket |F| > force_n for tilt_consec consecutive steps
  STUCK : plug-socket |F| > force_n AND peg xy error > stuck_xy_m for stuck_consec consecutive steps
  expert: LIFT (hold xy, ee z += lift_tilt_m / lift_stuck_m) -> ROTATE+ALIGN (peg xy servo onto the socket, plug
          orientation servo, z held) -> HANDBACK. Same gains/caps/end conditions as generate_tiltrec_demos phases 1-2.
          No scripted descent: after the alignment the policy descends itself.
  executed action = expert + N(0, noise) clipped to [-1, 1] (generator convention); that executed action is the label.

Triggers are evaluated on the state before the action (as in the generator); no trigger after the env has succeeded,
at most max_interventions per env, and none within cooldown steps after a handback.

Continuation (2026-10-10): after the handback the segment keeps recording the policy's own executed actions (phase 3)
for up to cont_max frames, stopping early at the next trigger or when the env has succeeded. Without it most STUCK
segments (median 25 frames) are shorter than the 24 frames a 32-step training chunk needs (horizon 32, pad_after 7,
pad_before 1) and give no training sample at all.

Profile r3full (dagger_r3full_20261011/criteria.json): the segment also stores the context_pre policy frames before the
trigger (phase 0, observation history only), a completed alignment continues into a scripted DESCEND (phase 4, = step2
R3: z action descend_z, xy + rotation servo continue, z 0 once success) and the segment ends at the first success
frame. No handback continuation (cont_max 0), so no policy-executed action after the trigger is a label. An alignment
timeout hands back to the policy with the segment closed incomplete. Without the profile nothing changes.

Profile r3full_v2 (dagger_r3full_v2_20261014/criteria.json): as r3full, but the env success flag (mean keypoint error
< 8.99 mm = socket depth, i.e. plug just inside the socket mouth) no longer stops the descent. DESCEND keeps z at
descend_z until the plug is fully inserted (plug_z - socket_z < deep_dz_m) and the segment ends at that frame; descend_z
matches the human-demo mean |z| (0.29 vs 0.15 in r3full).
"""
import collections
import json
import pathlib

import numpy as np

PARAMS = dict(takeover_kinds=['TILT', 'STUCK'], tilt_deg=10.0, force_n=1.0, tilt_consec=2, stuck_xy_m=0.002, stuck_consec=10,
              lift_tilt_m=0.020, lift_stuck_m=0.010, lift_tol_m=0.001, max_lift=20,
              rot_done_deg=2.0, align_tol_m=0.0007, align_ok_steps=3, max_align=80,
              kp_xy=0.6, kp_rot=0.5, pos_scale=0.01, rot_scale=0.05, rot_cap=0.3,
              lift_xy_cap=0.3, lift_z_cap=0.5, align_xy_cap=0.5, align_z_cap=0.3,
              noise=0.03, max_interventions=2, cooldown=5, cont_max=24)
PROFILES = {'r2': {},
            'r3full': dict(profile='r3full', context_pre=8, cont_max=0, descend_z=-0.15, max_descend=80),
            'r3full_v2': dict(profile='r3full_v2', context_pre=8, cont_max=0, descend_z=-0.29, max_descend=120,
                              deep_dz_m=0.001)}


def quat_mul(a, b):
    """Hamilton product, IsaacGym (x, y, z, w) convention (= generate_scripted_demos.quat_mul)."""
    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], axis=-1).astype(np.float32)


def quat_error_axis_angle(q_cur, q_goal):
    """World-frame axis-angle taking q_cur to q_goal (= generate_scripted_demos.quat_error_axis_angle)."""
    q_cur = q_cur / np.linalg.norm(q_cur, axis=-1, keepdims=True).clip(1e-8)
    q_goal = q_goal / np.linalg.norm(q_goal, axis=-1, keepdims=True).clip(1e-8)
    x1, y1, z1, w1 = q_goal[:, 0], q_goal[:, 1], q_goal[:, 2], q_goal[:, 3]
    x2, y2, z2, w2 = -q_cur[:, 0], -q_cur[:, 1], -q_cur[:, 2], q_cur[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    vec = np.stack([x, y, z], axis=-1)
    sin_half = np.linalg.norm(vec, axis=-1, keepdims=True)
    w = np.clip(np.abs(w), 0.0, 1.0)[:, None]
    angle = 2.0 * np.arctan2(sin_half, w)
    return (vec / sin_half.clip(1e-8) * angle).astype(np.float32)


def plug_orientation_error(plug_quat, socket_quat):
    """= generate_scripted_demos.plug_orientation_error (peg is 180 deg symmetric; pick the smaller correction)."""
    rz180 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    cand = [socket_quat, quat_mul(socket_quat, np.broadcast_to(rz180, socket_quat.shape))]
    errs = [quat_error_axis_angle(plug_quat, c) for c in cand]
    pick = (np.linalg.norm(errs[1], axis=-1) < np.linalg.norm(errs[0], axis=-1))[:, None]
    return np.where(pick, errs[1], errs[0])


def plug_tilt_deg(plug_quat):
    q = plug_quat / np.linalg.norm(plug_quat, axis=-1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    zz = 1 - 2 * (x * x + y * y)                       # world z component of the plug z axis
    return np.degrees(np.arccos(np.clip(np.abs(zz), 0, 1)))


class DaggerController:
    def __init__(self, n_envs, seed=0, params=None, record=True):
        self.p = dict(PARAMS, **(params or {}))
        self.n = n_envs
        self.rng = np.random.default_rng(seed)
        self.record = record
        self.mode = np.zeros(n_envs, int)              # 0 policy, 1 lift, 2 rotate+align
        self.k = np.zeros(n_envs, int); self.ok = np.zeros(n_envs, int)
        self.run_tilt = np.zeros(n_envs, int); self.run_stuck = np.zeros(n_envs, int)
        self.n_int = np.zeros(n_envs, int); self.cool_until = np.full(n_envs, -1)
        self.succeeded = np.zeros(n_envs, bool)
        self.goal = np.zeros((n_envs, 3)); self.xy_hold = np.zeros((n_envs, 2))
        self.cur = [None] * n_envs                     # open segment per env
        self.cont = [None] * n_envs                    # closed segment still recording its post-handback continuation
        self.segments = []                             # closed segments
        self.mode_log = []
        self.r3 = self.p.get('profile') in ('r3full', 'r3full_v2')   # mode 4 = DESCEND (r3full / r3full_v2)
        self.v2 = self.p.get('profile') == 'r3full_v2'                # descend to full insertion, not to the success flag
        self.ctx = [collections.deque(maxlen=int(self.p.get('context_pre', 0))) for _ in range(n_envs)]

    # ------------------------------------------------------------------ core
    def step(self, t, policy_action, priv, frame=None):
        """policy_action (N,6); priv: dict ee_pos, plug_pos, plug_quat, socket_pos, socket_quat, force (N,3),
        success (N,) bool at the current (pre-action) state; frame: per-env observation fields to record."""
        p = self.p
        action = np.array(policy_action, dtype=np.float32, copy=True)
        ee, pp, sp = priv['ee_pos'], priv['plug_pos'], priv['socket_pos']
        self.succeeded |= np.asarray(priv['success']).astype(bool)
        tilt = plug_tilt_deg(priv['plug_quat'])
        fmag = np.linalg.norm(priv['force'], axis=-1)
        err_xy = sp[:, :2] - pp[:, :2]; exy = np.linalg.norm(err_xy, axis=-1)
        contact = fmag > p['force_n']
        self.run_tilt = np.where(contact & (tilt > p['tilt_deg']), self.run_tilt + 1, 0)
        self.run_stuck = np.where(contact & (exy > p['stuck_xy_m']), self.run_stuck + 1, 0)
        rot_err = plug_orientation_error(priv['plug_quat'], priv['socket_quat'])
        deep = (pp[:, 2] - sp[:, 2]) < p['deep_dz_m'] if self.v2 else None
        reached = deep if self.v2 else self.succeeded                       # DESCEND end condition
        for e in range(self.n):
            if self.mode[e] == 0:
                can = (not self.succeeded[e]) and self.n_int[e] < p['max_interventions'] and t > self.cool_until[e]
                kind = None
                if can and self.run_tilt[e] >= p['tilt_consec']:
                    kind = 'TILT'
                elif can and self.run_stuck[e] >= p['stuck_consec']:
                    kind = 'STUCK'
                c = self.cont[e]
                if c is not None:
                    stop = 'trigger' if kind is not None else 'success' if self.succeeded[e] else \
                        'max' if c['n_cont'] >= p['cont_max'] else None
                    if stop is not None:
                        if stop == 'trigger':                                # drop the frames of the run that led to
                            run = int(self.run_tilt[e] if kind == 'TILT' else self.run_stuck[e])   # the re-trigger
                            cut = min(run, c['n_cont'])                      # (= pressing while stuck, bad labels)
                            if cut:
                                del c['actions'][-cut:]; del c['phases'][-cut:]
                                if c['frames']:
                                    del c['frames'][-cut:]
                                c['n_cont'] -= cut; c['cont_trimmed'] = cut
                        c['cont_end'] = stop; self.cont[e] = None
                if kind is not None and kind not in p['takeover_kinds']:   # r2tilt: detected (ends a continuation) but no takeover
                    kind = None
                if kind is not None:
                    self.mode[e], self.k[e], self.ok[e] = 1, 0, 0
                    lift = p['lift_tilt_m'] if kind == 'TILT' else p['lift_stuck_m']
                    self.goal[e] = [ee[e, 0], ee[e, 1], ee[e, 2] + lift]; self.xy_hold[e] = ee[e, :2]
                    self.cur[e] = {'env': e, 'kind': kind, 't_trigger': int(t), 'tilt_trig': float(tilt[e]),
                                   'xy_trig_mm': float(exy[e] * 1000), 'force_trig': float(fmag[e]),
                                   'ee_z_trig': float(ee[e, 2]), 'intervention_index': int(self.n_int[e]),
                                   'frames': [], 'actions': [], 'phases': []}
                    if self.r3:                                              # pre-trigger policy frames (phase 0)
                        self.cur[e]['context_len'] = 0
                        for f_, a_ in self.ctx[e]:
                            if f_ is not None:
                                self.cur[e]['frames'].append(f_)
                            self.cur[e]['actions'].append(a_); self.cur[e]['phases'].append(0)
                        self.cur[e]['context_len'] = len(self.ctx[e]); self.ctx[e].clear()
            if self.mode[e] == 0:
                if self.r3 and self.ctx[e].maxlen:                           # policy-driven this step -> context ring
                    self.ctx[e].append(({k: np.array(v[e], copy=True) for k, v in frame.items()}
                                        if (self.record and frame is not None) else None, action[e].copy()))
                continue
            a = np.zeros(6, np.float32)
            phase = int(self.mode[e])
            if self.mode[e] == 1:                                            # LIFT
                a[:2] = np.clip((self.xy_hold[e] - ee[e, :2]) / p['pos_scale'], -p['lift_xy_cap'], p['lift_xy_cap'])
                a[2] = np.clip((self.goal[e, 2] - ee[e, 2]) / p['pos_scale'], -p['lift_z_cap'], p['lift_z_cap'])
                self.k[e] += 1
                if abs(self.goal[e, 2] - ee[e, 2]) < p['lift_tol_m'] or self.k[e] >= p['max_lift']:
                    self.cur[e]['lift_steps'] = int(self.k[e]); self.mode[e], self.k[e] = 2, 0
            elif self.mode[e] == 4:                                           # DESCEND (r3full): servo xy/rot, z down
                a[:2] = np.clip(p['kp_xy'] * err_xy[e] / p['pos_scale'], -p['align_xy_cap'], p['align_xy_cap'])
                a[2] = 0.0 if reached[e] else p['descend_z']
                r = p['kp_rot'] * rot_err[e] / p['rot_scale']; nr = np.linalg.norm(r)
                a[3:] = r * min(1.0, p['rot_cap'] / max(nr, 1e-9)); self.k[e] += 1
            else:                                                             # ROTATE + ALIGN (peg xy servo)
                a[:2] = np.clip(p['kp_xy'] * err_xy[e] / p['pos_scale'], -p['align_xy_cap'], p['align_xy_cap'])
                a[2] = np.clip((self.goal[e, 2] - ee[e, 2]) / p['pos_scale'], -p['align_z_cap'], p['align_z_cap'])
                r = p['kp_rot'] * rot_err[e] / p['rot_scale']; nr = np.linalg.norm(r)
                a[3:] = r * min(1.0, p['rot_cap'] / max(nr, 1e-9)); self.k[e] += 1
                good = np.linalg.norm(rot_err[e]) < np.radians(p['rot_done_deg']) and exy[e] < p['align_tol_m']
                self.ok[e] = self.ok[e] + 1 if good else 0
            if p['noise'] > 0:
                a = a + self.rng.normal(0, p['noise'], 6).astype(np.float32)
            a = np.clip(a, -1, 1).astype(np.float32)
            action[e] = a
            seg = self.cur[e]
            if self.record and frame is not None:
                seg['frames'].append({k: np.array(v[e], copy=True) for k, v in frame.items()})
            seg['actions'].append(a.copy()); seg['phases'].append(phase)
            if self.mode[e] == 2 and (self.ok[e] >= p['align_ok_steps'] or self.k[e] >= p['max_align']):
                done = bool(self.ok[e] >= p['align_ok_steps'])
                seg.update(align_steps=int(self.k[e]), completed=done,
                           xy_after_mm=float(exy[e] * 1000), rot_err_after_deg=float(np.degrees(np.linalg.norm(rot_err[e]))),
                           t_handback=int(t + 1))
                if self.r3 and done:                                         # r3full: continue into DESCEND
                    seg.update(align_completed=True, world_tilt_at_descend_deg=float(tilt[e]), t_handback=None)
                    self.mode[e], self.k[e] = 4, 0
                else:
                    if self.r3:
                        seg.update(align_completed=False, cont_end='align_timeout')
                    self._close(e)
            elif self.mode[e] == 4 and (reached[e] or self.k[e] >= p['max_descend']):
                seg.update(descend_steps=int(self.k[e]), success_in_segment=bool(self.succeeded[e]),
                           cont_end=('deep' if self.v2 else 'success') if reached[e] else 'descend_timeout', t_handback=int(t + 1))
                if self.v2:
                    seg.update(deep_in_segment=bool(reached[e]), end_dz_mm=float((pp[e, 2] - sp[e, 2]) * 1000))
                self._close(e)
        for e in range(self.n):                                             # post-handback continuation (policy labels)
            c = self.cont[e]
            if c is None or self.mode[e] != 0 or c.get('t_handback') is None or t < c['t_handback']:
                continue
            if self.record and frame is not None:
                c['frames'].append({k: np.array(v[e], copy=True) for k, v in frame.items()})
            c['actions'].append(action[e].copy()); c['phases'].append(3); c['n_cont'] += 1
        self.mode_log.append(self.mode.copy())
        return action

    def _close(self, e):
        self.cur[e]['n_expert'] = len(self.cur[e]['actions']) - self.cur[e].get('context_len', 0); self.cur[e]['n_cont'] = 0
        self.segments.append(self.cur[e]); self.cont[e] = None if self.r3 else self.cur[e]; self.cur[e] = None
        self.mode[e], self.k[e], self.ok[e] = 0, 0, 0
        self.n_int[e] += 1; self.cool_until[e] = self.segments[-1]['t_handback'] + self.p['cooldown']
        self.run_tilt[e] = 0; self.run_stuck[e] = 0

    # ------------------------------------------------------------------ end of run
    def finish(self, success_traj, out_dir, zarr_path=None, cohort=None):
        """success_traj (T, N) bool. Closes open segments (incomplete), fills post-handback outcome, writes
        dagger_log.json + (optionally) appends segments to a ReplayBuffer zarr and a jsonl sidecar."""
        T = len(success_traj)
        for e in range(self.n):
            if self.cur[e] is not None:
                self.cur[e].update(completed=False, t_handback=None, cut_by_episode_end=True,
                                   n_expert=len(self.cur[e]['actions']) - self.cur[e].get('context_len', 0), n_cont=0)
                if self.r3:
                    self.cur[e].update(cont_end='episode_end', success_in_segment=False)
                self.segments.append(self.cur[e]); self.cur[e] = None
        for e in range(self.n):
            if self.cont[e] is not None:
                self.cont[e]['cont_end'] = 'episode_end'; self.cont[e] = None
        S = np.asarray(success_traj).astype(bool)
        for s in self.segments:
            e = s['env']; th = s.get('t_handback')
            s['success_after_handback'] = bool(th is not None and th < T and S[th:, e].any())
            s['success_episode'] = bool(S[:, e].any())
            s['n_frames'] = len(s['actions'])
            if self.r3:                                                      # criteria kept_segment
                s['keep'] = bool(s['kind'] == 'TILT' and s.get('align_completed') and
                                 s.get('deep_in_segment' if self.v2 else 'success_in_segment'))
        info = [{k: v for k, v in s.items() if k not in ('frames', 'actions', 'phases')} for s in self.segments]
        out_dir = pathlib.Path(out_dir)
        (out_dir / 'dagger_log.json').write_text(json.dumps({'params': self.p, 'segments': info,
                                                              'n_interventions_per_env': self.n_int.tolist()}, indent=1))
        np.save(out_dir / 'dagger_mode.npy', np.stack(self.mode_log).astype(np.int8) if self.mode_log else np.zeros((0, self.n), np.int8))
        if zarr_path is None or not self.record:
            return info
        from diffusion_policy.common.replay_buffer import ReplayBuffer
        buf = ReplayBuffer.create_from_path(str(zarr_path), mode='a')
        side = open(str(zarr_path) + '_segments.jsonl', 'a')
        for s, d in zip(self.segments, info):
            if not s['frames']:
                continue
            ep = {k: np.stack([f[k] for f in s['frames']]) for k in s['frames'][0]}
            ep['action'] = np.stack(s['actions']).astype(np.float32)
            ep['phase'] = np.asarray(s['phases'], np.int8)
            ep['trigger_kind'] = np.full(len(s['actions']), 0 if s['kind'] == 'TILT' else 1, np.int8)
            d = dict(d, cohort=cohort, zarr_episode_index=int(buf.n_episodes))
            buf.add_episode(ep, compressors='disk')
            side.write(json.dumps(d) + '\n')
        side.close()
        return info
