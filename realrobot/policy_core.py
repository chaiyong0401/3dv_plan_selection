"""Real-robot policy core (2026-11-16).

One process holds the whole policy: generator (LDP + AT decoder), Q selector with the belief rule and the
continuity term, and the per-step decode. It re-uses the functions of the simulation server
(default policy_server_selrule_stallr_20261117.py = the server of the current simulation system Q400B / ER3; env
RDP_SERVER_MODULE selects another, e.g. policy_server_selrule_xfade_20261116 for the 11-16 validation) instead of
re-implementing them, so the selection is the same code
that produced the simulation results. What the server did per socket request is done here per call:

  plan(snapshot)          refresh: draw K candidates in one generator call, score them, apply the belief rule,
                          return the chosen latent.               (server: chunk_step 0 request)
  decode(latent, window)  per-step command of a held latent on the current tactile window.
                          (server: every request)
  push_frame / note_action  observation and history buffers.   (bridge buffers + server deques)

The core has no notion of time; runtime.py decides when a plan is requested, when it takes effect (latency
matching) and how the old and new plan are blended (cross-fade).

Robot-independent. robot_command_sent stays false until the user enables a real robot interface.
"""
import collections
import copy
import os
import sys

import numpy as np
import torch

RDP = '/home/dyros-recruit/mcy_ws/reactive_diffusion_policy'
if RDP not in sys.path:
    sys.path.insert(0, RDP)
os.environ.setdefault('RDP_DRAW_SINGLE', '1')          # all K candidates in one generator call (rt_speed2)
import importlib                                       # noqa: E402
S = importlib.import_module(os.environ.get('RDP_SERVER_MODULE', 'policy_server_selrule_stallr_20261117'))
# stallr with RDP_STALL_RATIO / RDP_QCONF / RDP_QZERO unset selects exactly as xfade_20261116 (diff checked 2026-11-19).

# Evaluation arms of the real-robot plan (qsel_core/REAL_ROBOT_MS_PLAN.md, 4-1). Residual on/off is the Runtime's
# `residual` argument; the arm only fixes the selection side.
#   q    : 'learned' (Q ensemble) | 'zero' (RDP_QZERO: first candidate the rule allows = random, candidates are i.i.d.)
#          | 'none' (no verifier: first candidate every refresh)
#   rule : True = stall-based blocking rule (evid), False = Q picks among all candidates every refresh
ARMS = {
    'A0_generator':          dict(q='none', rule=False, residual=False),
    'A1_generator_residual': dict(q='none', rule=False, residual=True),
    'A2_full':               dict(q='learned', rule=True, residual=True),
    'A3_no_corrector':       dict(q='learned', rule=True, residual=False),
    'A4_no_rule':            dict(q='learned', rule=False, residual=True),
    'A5_no_q':               dict(q='zero', rule=True, residual=True),
    'A6_full_er3':           dict(q='learned', rule=True, residual=True, er3=True),
}


DEFAULTS = dict(
    # task / at_config: the SIMULATION USB config (wrist + front camera). The real robot has no external camera:
    # the generator is retrained on wrist-only data and its own config goes here; obs keys then come from the
    # checkpoint (self.obs_keys), nothing else in this file reads 'front'.
    task='manifeel_pih_c8d_ldp_front_aligned', at_config='at_manifeel_pih_l16', device='cuda',
    select_k=16, select_horizon=24, latent_refresh=12,
    no_clip_sample=True,
    coh=dict(lam=0.02, m=4, d=2),                      # continuity term (Q - lam * mismatch of steps d..d+m-1)
    # Current system (11-17, Q400B / ER3): blocking only = --evid-k 999 --evid-suppress --evid-fallback 2 (switch
    # candidates are held back until 2 consecutive stalls; no promotion). None disables the rule (arm A4).
    evid=dict(k=999, perm_k=-1, theta=1.0, suppress=True, fallback=2),
    # 'stall' = the hand did not move. 'ratio' (default, unit-free, 2026-11-16): 24-step displacement / commanded
    # displacement over the same 24 steps < ratio (0.148 matches the 3 mm stall rate on the USB logs; the belief table
    # keeps its shape: 0.875 / 0.962 / 1.00 at evidence 1 / 2 / 3). 'mm': 24-step displacement < mm metres (the
    # simulation runs up to 2026-11-16; use it to reproduce them).
    # 2026-11-19: default back to 'mm' 3 mm = the stall of every current simulation result. The ratio stall with the
    # data-derived 0.0547 lost 7 pp vs 3 mm (evrel_20261117 BLOCKR vs BLOCK, 1 seed). On the real robot the threshold
    # is re-derived from rollouts with log-scale Otsu of the 24-step displacement (USB sim: 2.5 / 2.9 mm on dev /
    # holdout, stallabs_20261119) before evaluation.
    stall=dict(mode='mm', ratio=0.148, mm=0.003),
    q='learned',                                       # 'learned' | 'zero' | 'none' (see ARMS)
    # promotion count k: smallest evidence count whose share of wrong-mode episodes in policy rollouts is >= 0.95
    # (calibrate_belief.choose_k); the choice is k = 2 for any criterion in 0.85 .. 0.99 on the USB logs.
    q_start_step=36,                                   # the server builds the Q state only from step 36
)


class PolicyCore:
    def __init__(self, ldp_ckpt, at_ckpt, verifier_ckpts, evid_est=None, residual_ckpt=None, n_envs=1, **over):
        cfg = copy.deepcopy(DEFAULTS)
        arm = over.pop('arm', None)
        if arm is not None:
            a = ARMS[arm]; cfg['q'] = a['q']
            if not a['rule']:
                cfg['evid'] = None
        for k, v in over.items():
            cfg[k] = v
        if cfg['q'] == 'none':
            verifier_ckpts = None
        if cfg['q'] == 'zero':
            os.environ['RDP_QZERO'] = '1'                 # process-wide: one PolicyCore per process for this arm
        elif os.environ.get('RDP_QZERO') == '1':
            raise RuntimeError('RDP_QZERO=1 is set in this process but q != zero')
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        self.policy, self.constants = S.load_policy(ldp_ckpt, at_ckpt, cfg['task'], cfg['device'], cfg['at_config'],
                                                    extended_obs_mode='growing')
        if cfg['no_clip_sample']:
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            self.policy.noise_scheduler = DDIMScheduler.from_config({**self.policy.noise_scheduler.config, 'clip_sample': False})
        c = self.constants
        self.obs_steps = int(c['dataset_obs_steps']); self.ratio = int(c['obs_temporal_downsample_ratio'])
        self.horizon = int(c['original_horizon']); self.obs_keys = list(c['obs_keys']); self.ext_keys = list(c['extended_obs_keys'])
        S.COH.update(lam=float(cfg['coh']['lam']), m=int(cfg['coh']['m']), d=int(cfg['coh']['d']), prev=None)
        assert S.COH['R'] + S.COH['d'] + S.COH['m'] <= cfg['select_horizon']
        S.EVID = None
        if evid_est is not None and cfg['evid'] is not None:
            e, dv = np.load(evid_est), cfg['device']; kk = torch.as_tensor(e['k'], device=dv)
            t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dv)
            ev = cfg['evid']
            S.EVID = {'k': kk, 'mu': t(e['mu'])[kk], 'sd': t(e['sd'])[kk], 'wx': t(e['wx']), 'bx': float(e['bx']), 'wr': t(e['wr']), 'br': float(e['br']),
                      'kstar': int(ev['k']), 'perm_k': int(ev['perm_k']) if int(ev['perm_k']) >= 0 else int(ev['k']), 'theta': float(ev['theta']),
                      'suppress': bool(ev['suppress']), 'fallback': int(ev['fallback']), 'soft': 0.0, 'force_cs': 10 ** 9, 'ext': False}
        self.verifier = S.load_verifier(verifier_ckpts, cfg['device']) if verifier_ckpts else None
        self.residual_ckpt = residual_ckpt
        self.n = int(n_envs)
        self.reset()

    # ------------------------------------------------------------------ episode state
    def reset(self, n_envs=None):
        if n_envs is not None:
            self.n = int(n_envs)
        h = self.cfg['select_horizon']
        self.obs_buf = [collections.deque(maxlen=self.obs_steps) for _ in range(self.n)]
        self.ext_buf = [collections.deque(maxlen=self.horizon) for _ in range(self.n)]
        self.recent_actions = collections.deque(maxlen=h)   # decoded policy commands (before the residual), as the server
        self.recent12 = collections.deque(maxlen=12)
        self.z_history = collections.deque(maxlen=64)
        self.pose_history = collections.deque(maxlen=64)
        self.step = 0
        S.EVID_STATE.clear()
        S.COH['prev'] = None
        self.refresh_idx = -1
        self.feats_now = None

    def push_frame(self, fields, fill=False):
        """fields: dict key -> array (N, ...) for every key the policy reads (wrist/front float 0..255 CHW as in the
        simulation bridge, state (N,7), contact_8d_right (N,8)). fill=True at episode start (repeats the frame)."""
        for i in range(self.n):
            f = {k: np.asarray(fields[k][i]) for k in set(self.obs_keys) | set(self.ext_keys)}
            for _ in range(self.obs_steps if fill else 1):
                self.obs_buf[i].append(f); self.ext_buf[i].append(f)

    def _obs_dict(self):
        out = {k: [] for k in self.obs_keys}
        for i in range(self.n):
            sel = list(self.obs_buf[i])[::-self.ratio][::-1]
            for k in self.obs_keys:
                out[k].append(np.stack([f[k] for f in sel]))
        return S.to_torch({k: np.stack(v) for k, v in out.items()}, self.device)

    def ext_window(self, n_frames):
        n_frames = int(min(n_frames, self.horizon))
        out = {k: [] for k in self.ext_keys}
        for i in range(self.n):
            fr = list(self.ext_buf[i])[-n_frames:]
            for k in self.ext_keys:
                out[k].append(np.stack([f[k] for f in fr]))
        return S.to_torch({k: np.stack(v) for k, v in out.items()}, self.device), n_frames

    # ------------------------------------------------------------------ per step
    @torch.no_grad()
    def step_features(self):
        """obs_features of the current observation (the residual reads them every step)."""
        self.feats_now = S.obs_features(self.policy, self._obs_dict()).half().cpu().numpy()
        return self.feats_now

    @torch.no_grad()
    def decode(self, latent, steps_since_plan_obs):
        """Command of a held plan now. The window has obs_steps + (steps since the plan's observation) frames,
        i.e. the plan continues from where it would be had it been executed from its own observation on."""
        ext, n = self.ext_window(self.obs_steps + int(steps_since_plan_obs))
        a = self.policy.predict_from_latent_action(latent, ext, n, self.ratio)['action'][:, -1]
        return a

    def note_action(self, policy_action, state):
        """After every executed step: the decoded (blended) policy command and the observed state of that step."""
        a = torch.as_tensor(policy_action, device=self.device, dtype=torch.float32)
        st = torch.as_tensor(np.asarray(state, np.float32), device=self.device)
        self.recent_actions.append(a.detach()); self.recent12.append(a.detach())
        self.z_history.append(st[:, 2].clone()); self.pose_history.append(st[:, :3].clone())
        self.step += 1

    # ------------------------------------------------------------------ refresh
    def snapshot(self):
        """Everything plan() needs, copied, so a planner thread can work while the control loop goes on."""
        obs = self._obs_dict()
        ext, n = self.ext_window(self.obs_steps)
        return dict(step=self.step, obs=obs, ext=ext, recent_actions=collections.deque(self.recent_actions, maxlen=self.recent_actions.maxlen),
                    recent12=list(self.recent12), z_history=list(self.z_history), pose_history=list(self.pose_history))

    def _stall(self, Pg, snap):
        sc = self.cfg['stall']
        if sc['mode'] == 'mm':
            return Pg[:, 10].float() < float(sc['mm'])
        acts = torch.stack(list(snap['recent_actions'])[-24:], 1)                  # (N, <=24, A) policy commands
        cmd24 = acts[..., :3].sum(1).norm(dim=-1) * 0.01                           # commanded displacement [m]
        return (Pg[:, 10].float() / cmd24.clamp_min(1e-6)) < float(sc['ratio'])

    @torch.no_grad()
    def plan(self, snap, seed=None):
        """Same computation as the server's chunk_step-0 request with select_rule learned_q. Returns (latent, info)."""
        if seed is not None:
            torch.manual_seed(int(seed))
        obs, ext, step = snap['obs'], snap['ext'], snap['step']
        feats = S.obs_features(self.policy, obs) if self.verifier is not None else None
        vstate = None
        if self.verifier is not None and len(snap['recent12']) == 12 and step >= self.cfg['q_start_step'] and len(snap['z_history']) >= 24:
            stt = obs['state'][:, -1]
            acts = torch.stack(snap['recent12'], 1); astd = (acts - acts.mean(1, keepdim=True)).pow(2).mean(1).sqrt()
            tcol = torch.full((stt.shape[0], 1), step / 499.0, device=stt.device)
            ph, zh = snap['pose_history'], snap['z_history']
            P = torch.cat([stt[:, :7], (zh[-24] - stt[:, 2])[:, None], acts.mean(1), astd, tcol], 1)
            d24, d12 = stt[:, :3] - ph[-24], stt[:, :3] - ph[-12]
            Pg = torch.cat([stt[:, :7], d24, d24.norm(dim=-1, keepdim=True), d12, d12.norm(dim=-1, keepdim=True), acts.mean(1), astd, tcol], 1)
            vstate = {'obs_features': feats.float(), 'P': P.float(), 'P_generic': Pg.float()}
            if S.EVID is not None:
                E, ES = S.EVID, S.EVID_STATE
                x_ = torch.cat([feats.half().float(), stt[:, :7].float()], 1)
                z_ = ((x_[:, E['k']] - E['mu']) / E['sd']).clamp(-10, 10)
                px_ = z_ @ E['wx'] + E['bx']; pr_ = z_ @ E['wr'] + E['br']
                stl = self._stall(Pg, snap)
                ok_ = stl & (px_ < 0.5) & (pr_ < 2.9)
                if 'c' not in ES:
                    ES['c'] = torch.zeros_like(px_, dtype=torch.long); ES['done'] = torch.zeros_like(ok_)
                ES['c'] = torch.where(ok_, ES['c'] + 1, torch.zeros_like(ES['c']))
                st_ = stl
                ES['cs'] = torch.where(st_, ES.get('cs', torch.zeros_like(ES['c'])) + 1, torch.zeros_like(ES['c']))
                vstate['evid'] = {'c': ES['c'].clone(), 'cs': ES['cs'].clone(), 'px': px_, 'pr': pr_}
        self.refresh_idx += 1
        rule = 'learned_q' if self.verifier is not None else 'first'
        latent, idx, info = S.select_latent(self.policy, self.constants, obs, ext, rule, self.cfg['select_k'], self.cfg['select_horizon'],
                                            snap['recent_actions'], z_history=snap['z_history'], rng=np.random.default_rng(0 if seed is None else int(seed)),
                                            verifier=self.verifier, verifier_state=vstate, refresh_idx=self.refresh_idx)
        keep = {k: (v.detach().float().cpu().numpy() if torch.is_tensor(v) else v) for k, v in info.items()
                if k in ('dist', 'verifier_Q', 'evid_c', 'evid_cs', 'evid_sup', 'evid_force', 'coh')}
        keep['chosen'] = idx.detach().cpu().numpy(); keep['step'] = step
        return latent.detach(), keep
