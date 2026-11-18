"""[b2_20261117: every reply also carries evid_done (N,) = env has chosen a switch candidate (dist > theta) in this episode, so the bridge can switch the residual after the first switch. Selection unchanged.] [stallr_20261117: env RDP_STALL_RATIO=T -- stall = 24-step displacement / (0.01 * ||sum of the last 24 sent actions xyz||) < T instead of < 3 mm, for both counters; records evid_perm. Unset = identical to the source] [qconf_20261117: env RDP_QCONF=1 -- a switch candidate (dist > theta) chosen by Q outside the evidence promotion is kept only if EVERY ensemble member prefers it to the best stay candidate (advantage incl. continuity term); otherwise the best stay candidate is taken. Unset = identical to the source] [qlow_rule_20261116: env RDP_QZERO=1 replaces the Q scores by -1000*index, so selection = first candidate allowed by the belief rule (no Q); unset = identical to the source] [belief_parts_20261116: + --evid-perm-k, evidence count that lifts the switch suppression separately from the promotion count --evid-k; default unchanged] policy_server_selrule_rt_20261113.py (rt_final_20261113) = policy_server_selrule_coh_20261113.py + execution-time selection (after Real-Time EXPO-FT): a request carrying pregen draws the K candidates from THAT request's observation in one generator call and stores them; a later refresh request carrying use_pregen skips drawing and scores the stored candidates with the CURRENT observation features, decoding them from the tactile window the bridge sends (longer by the pregen lead, so the decoded plans start at the step that will be executed). Without those request fields nothing changes. Original docstring: policy_server_selrule_coh_20261113.py (prepared 2026-11-13, not yet used in a registered run) = policy_server_selrule_soft_20261113.py + continuity term: --coh-lambda L (0 = off, unchanged), --coh-steps m, --coh-delay d. Selection uses Q - L * ||sum(new plan xyz steps d..d+m-1) - sum(previous chosen plan xyz steps R+d..R+d+m-1)|| (raw action units, R = 12), i.e. how well the head of each candidate continues what the previous plan would have done over the same time steps. The recorded Q is unchanged. Original docstring: policy_server_selrule_soft_20261113.py (soft_belief_20261113) = policy_server_selrule_speed2_20261113.py + --evid-soft lambda (default 0 = unchanged). Original docstring: policy_server_selrule_speed2_20261113.py (rt_speed2_20261113) = policy_server_selrule_speed_20261113.py + env RDP_DRAW_SINGLE=1: all K candidates are generated in one generator call (default off = unchanged). Original docstring: policy_server_selrule_speed_20261113.py (rt_speed_20261113) = policy_server_selrule_delay_20261113.py + env RDP_INFER_STEPS overriding the DDIM inference step count (default 8, unchanged). Original docstring: policy_server_selrule_delay_20261113.py (new 2026-11-13; rt_robust_20261113) = policy_server_selrule_evidext_20261111.py + execution-delay emulation driven by the bridge: a refresh request carrying delay_base_n computes the new latent from the last delay_base_n tactile frames as usual, but the PREVIOUS latent keeps being decoded (with the long window the bridge sends) until a request carrying delay_swap arrives; then the new latent takes over at its own chunk step. Without those request fields the behaviour is unchanged. Original docstring: policy_server_selrule_evidext_20261111.py (new 2026-11-11; link_cl_20261111) = policy_server_selrule_evidthr_20261111.py + --evid-ext: the evidence inputs px / pr come from the bridge (req[evid_ext] = residual head margins; + 0.5 / + 2.9 so the unchanged thresholds apply).
policy_server_selrule_evidthr_20261111.py (new 2026-11-11; q_thr_20261111) = policy_server_selrule_evid2_20261110.py + --evid-force-cs m: the first switch is also forced when the plain stall counter cs >= m (default off = identical behaviour).
policy_server_selrule_evid2_20261110.py (new 2026-11-10; evid2_cl_20261110) = policy_server_selrule_evid_20261110.py + --evid-suppress / --evid-fallback m:
with --evid-suppress, while the evidence counter c < k AND the plain stall counter cs (consecutive refreshes with d24 < 3 mm) < m, the argmax of Q is restricted to the
NON-switch candidates (dist <= theta) if any exist; promotion (c >= k, first switch only) is unchanged. Records add evid_cs, evid_sup (1 where the restriction changed the choice).
Original: policy_server_selrule_evid_20261110.py (new 2026-11-10; evidence_rule_20261110): copy of policy_server_selrule_decu_noclip_20261107.py
with ONE addition: --evid-est <npz> enables the evidence-promoted switch on top of the learned Q.
Per connection (= episode batch), at every refresh where the Q state is built (step >= 36): alignment estimate from
x = [obs_features.half().float() (2078), ee pose (7)] with the ridge in the npz (pred xy mm, pred rot deg); evidence counter
c = c + 1 if (24-step EE displacement d24 < 3 mm AND pred xy < 0.5 AND pred rot < 2.9) else 0.
Selection (rule learned_q only): switch candidates = recorded dist > --evid-theta. If c >= --evid-k, the env has switch candidates and has
never chosen a switch candidate before in this episode, the argmax of Q is restricted to the switch candidates (promotion); otherwise the
argmax of Q over all candidates is unchanged. Once an env has chosen a switch candidate (by Q or by promotion) promotion is disabled for it.
Records: evid_c, evid_px, evid_pr, evid_force, evid_done per verifier refresh. Without --evid-est nothing changes.
(original docstring follows)"""
EVID = None   # set in main: dict(mu, sd, k, wx, bx, wr, br tensors, kstar, theta)
EVID_STATE = {}   # per connection: c (N,), done (N,) bool


"""policy_server_selrule_decu_noclip_20261107.py (new 2026-11-07; usb_keyed_s13a): copy of policy_server_selrule_decu_20261024.py with ONE
addition: --no-clip-sample (DDIM clip_sample false at inference, code identical to policy_server_selrule_hazmem_fp_feats_20261106.py),
needed because F13/F15 are always served without clipping. Everything else (learned_q, records, want_feats) unchanged.
(original docstring follows)
policy_server_selrule_decu_20261024.py (new file 2026-10-24, copy of policy_server_selrule_rl_20261023.py; multisocket_decu_20261024):
in rule dec_soft, where the bridge sends options['explore_k'][i] >= 0 the executed candidate is replaced by that uniform index after the
dec_soft choice (known propensity 1/16); records explore_k and explore_base_idx (the dec_soft choice that was replaced; = chosen elsewhere).
Without explore_k (or all -1) the rule is unchanged. Previous docstring follows."""
"""policy_server_selrule_rl_20261023.py (new file, 2026-10-23; copy of policy_server_selrule_dec_20261023.py + per-step features):
if a request carries want_feats=True, the reply also carries 'obs_features' (B, 2078) float16 = obs_features(policy, obs_dict,
feat_encoder) of THIS request's obs (the served LDP's own frozen obs normalizer + obs_encoder, exactly the features recorded at
refreshes), for the RL residual (delta_residual_ms_rl_20261023). Nothing else changes. Previous docstring follows."""
"""policy_server_selrule_dec_20261023.py (new file, 2026-10-23; copy of policy_server_selrule_haz_20261022.py + rule dec_soft).
multisocket_dec_sim_20261023: decomposed candidate verifier A_hat_k = V(x + f(x, u_k)) - V(x) (multisocket_decomp_verifier_offline_20261022,
V_obs variant; deploy artifacts --dec-dir: dec_model.npz (PCA64, x / u standardisation), dyn_{0,1,2}.pt (dynamics MLP ensemble), v_obs.pkl).
x = [PCA64(obs_features fp16), c, |c|, d_g(s-1)] with c, d_g, recent-contact flag sent by the bridge as options['dec_x'] (N, 5).
Per row: p = the soft_tilt distribution exactly as SOFT16; if d_g > 0 and the row has both generic retreat (p_k . c < 0) and press candidates,
class masses are set to pi_g(d_g) / 1 - pi_g(d_g) (--dec-curve hazard_curve_generic.json, within-class shape of p kept); if the recent-contact
flag is set, q ∝ p * exp(lambda * A_hat) (class masses restored on hazard rows), lambda in [0, 1000] by bisection so that KL(q || p) <= --dec-eps;
execution by maximal coupling from candidate 0 with the SOFT16 rng (soft_seed, seed, env) -> rows without hazard / contact choose as SOFT16.
Records dec_ahat (k, N), dec_haz, dec_contact, dec_kl, dec_moved, dec_p / dec_base / dec_psoft (N, k). Previous docstring follows."""
"""policy_server_selrule_haz_20261022.py (new file, 2026-10-22; copy of policy_server_selrule_explore_20261022.py + rules haz / haz_soft).
multisocket_haz_20261022: a memoryless generator's K natural samples are reweighted so that the executed probability of a stop-pressing
(lift) plan follows the demonstrators' stall-duration hazard pi_star(d) = sigmoid(a + b * min(d, d_cap)) (--haz-curve hazard_curve.json of
multisocket_demo_stall_hazard_20261022). Lift candidate = decoded 12-step plan z sum > 0. With stall_d[i] = 0, or no / only lift candidates,
nothing changes. The target is the KL-minimal reweighting of the uniform empirical distribution under the class-mass constraint:
  haz      : mass pi_star on the lift class and 1 - pi_star on the press class, uniform within each class;
  haz_soft : the same class masses, within each class softmax(lmax * A_hat) with the soft_tilt critic advantage A_hat (same mode space /
             critic as soft_tilt; all-zero A_hat -> uniform).
Execution keeps candidate 0 with probability min(1, k p_0) and otherwise samples the residual (maximal coupling, rng (soft_seed, seed, env)).
Records haz_d, haz_target, haz_nlift, haz_moved, haz_soft_moved, haz_ahat (k, N critic advantage used; zeros for haz).
2026-10-22 (post-reboot, before freezing): in haz_soft, rows where the hazard is not applicable (stall_d = 0, no / only lift candidates)
now run soft_tilt exactly as SOFT16 (KL budget eps0 * mode_s, rng (soft_seed, seed, env)) instead of candidate 0, so HAZSOFT16 differs
from SOFT16 only in hazard rows (as HAZ16 differs from B16 only there). Original docstring follows."""
"""policy_server_selrule_explore_20261022.py (new file, 2026-10-22; copy of policy_server_selrule_rimorc_20261022.py + rule soft_explore).
soft_explore (multisocket_cand_q_explore_20261022, data collection for a candidate-level Q): soft_tilt exactly as before, then for env
where the bridge sends options['explore_k'][i] >= 0 the executed candidate is replaced by that uniformly drawn index (known propensity
1/16). Records explore_k and explore_soft_idx (the soft choice that was replaced). Original docstring follows."""
"""policy_server_selrule_rimorc_20261022.py (new file, 2026-10-22; copy of policy_server_selrule_soft_20261021.py + rule soft_rimorc).
soft_rimorc (multisocket_rim_oracle_ceiling_20261022, privileged CEILING only): soft_tilt exactly as before, then for env the bridge marks
as on the fit-socket rim (options['rim_orc'] (N, 3) = [flag, dx_mm, dy_mm], plug bottom - fit slot centre), the executed candidate is
replaced by argmin_k | (dx, dy) + 1000 * 0.01 * sum_h plan_raw[k, :, h, :2] | (the candidate whose 12-step lateral endpoint best aligns
the plug with the hole). Records rim_flag and rim_override. Original docstring follows."""
"""policy_server_selrule_soft_20261021.py (new file, 2026-10-21; copy of policy_server_selrule_mode_20261021.py + rule soft_tilt).
soft_tilt (multisocket_soft_tilt_sim_20261021): continuous-weight form of mode_first. Over the K natural samples, pi_k proportional to
exp(lambda A_hat(z_k)), A_hat = sum_c q_c(z) max(tau_pess_c, 0) with isotropic-GMM soft membership q (--mode-space npz must carry share,
sigma2, eps0, lmax), lambda by bisection so that KL(pi || U_K) = eps0 * s (s = the bridge's continuous stuck score mode_s, 0 outside
support). Execution: maximal coupling with candidate 0 (keep it w.p. min(1, K pi_0), else draw from (pi - 1/K)+), uniforms from
numpy default_rng([soft_seed, request policy_seed, env]). Records mode_lab, mode_tau (tau_pess), mode_dev (moved), soft_kl, soft_pi.
Identical to multisocket_soft_tilt_v2_20261021 stage1_offline.py tilt(). Original docstring follows."""
"""policy_server_selrule_mode_20261021.py (new file, 2026-10-21; copy of policy_server_selrule_stop_20261021.py + rule mode_first).
mode_first (multisocket_mode_eval_20261021): candidates are assigned to unsupervised plan modes with --mode-space (npz: per-dim sd, PCA
mean/components, k-means centers; multisocket_stuck_mode_offline_v2_20261021): feature = decoded 12x6 plan / sd, flattened, PCA, nearest
center. With the bridge's mode_mu (N, NB, k) and mode_gate (N,): P_c = (n_c among candidates 1..K-1... i.e. all but candidate 0 + 0.5) /
(K - 1 + 0.5 k); tau_c = mu_c - sum_c' w_c' mu_c' / sum w (w = P on modeled modes), pessimistic mean - std over members; among modes
present in the K candidates choose c* = argmax; where gated and tau_pess(c*) > 0 execute the first candidate of mode c*, else candidate 0
(identical to the offline policy of multisocket_stuck_mode_offline_v2). Records mode_lab, mode_tau, mode_dev. Original docstring follows."""
"""policy_server_selrule_stop_20261021.py (new file, 2026-10-21; copy of policy_server_selrule_guide_20261020.py + rule stop_first).
stop_first (multisocket_stop_eval_20261021): a candidate is STOP if its decoded 12-step plan z-sum * 10 mm > STOP_MM (-5). Where the
bridge sends stop_flag[i] = 1 and a STOP candidate exists, the first STOP candidate in draw order is executed; elsewhere candidate 0
(= B16). No guidance. Original docstring follows."""
"""policy_server_selrule_guide_20261020.py (new file, 2026-10-20; copy of policy_server_selrule_term_20261020.py + rule term_guide).
term_guide: tau-weighted latent escape guidance. For envs with switch_tau > 0 (sent by the guide bridge = tau_pess where the
critic flag is set, else 0), the second half of the K candidates (k//2..k-1) are moved in latent space by GUIDE_STEPS normalized
gradient steps that increase the distance of their decoded (normalized) plan from the MEDIAN plan of the K unguided candidates,
in units of the candidate set's own spread sigma (median distance to the median), hinged at GUIDE_CAP sigma; trust region
||z - z0|| <= r ||z0|| with r = GUIDE_RMAX * clip(tau / GUIDE_TAU_REF, 0, 1). No task constant enters the objective.
Selection after guidance is term_switch's (first 15 mm SWITCH candidate for flagged envs, else candidate 0), so the only
change versus term_switch is the candidate supply. --guide-force (benchmark) treats every env as flagged with tau = TAU_REF;
--guide-measure-only computes the guided candidates but always executes candidate 0 (coverage pilot).
Original docstring follows."""
"""multisocket_term_critic_20261020 (new file 2026-10-20, copy of policy_server_selrule_qt_20261020.py): two task-agnostic rules.
A candidate is SWITCH if its decoded plan xy displacement |sum_h plan_raw[h, :2]| * 10 mm >= TERM_SW_MM (15), else STAY.
  rand_switch (collection): at refreshes with step >= 24 and the bridge qt_extra contact flag (col 22) set, with probability
               TERM_EPS (0.25) execute a uniformly drawn SWITCH candidate if any exists, else candidate 0 (= first).
               Draws use the per-request numpy rng (seeded by the policy seed), torch RNG untouched.
  term_switch (deploy): where the bridge sends switch_flag[i] = 1, execute the first SWITCH candidate in the base draw order
               (if none exists, candidate 0); elsewhere candidate 0. The flag comes from the bridge-side critic.
Everything else unchanged. Previous docstring follows."""
"""QT server (new file, 2026-10-20, multisocket_qtime_20261020): copy of policy_server_selrule_orc_20261019.py plus
(1) verifier feature set 'qt_v1': a --verifier-ckpt whose feature_set is 'qt_v1' (no align / d term, use_align False) is
scored with P = qt_extra[:, :37] (PiH P 21 + VMS H 14 + dwell 2), T = qt_extra[:, 37:53], stuck = qt_extra[:, 53], candidate
features vms_features_20261019.cand_features (13) and obs_features rounded through fp16 as in the recorded training data;
qt_extra (N, 54) comes from the bridge eval_rdp_bridge_multisocket_qt_20261020.py --send_qt_extra
(qt_features_20261020.QTHistory);
(2) rule 'dwell_rule' (no learning, no privileged input): in contact (qt H contact flag) with dwell < 48 in-contact steps
at the current region -> candidate with the smallest nominal plan xy displacement (stay); dwell >= 96 and EE z descent
since the region anchor < 1 mm -> the largest displacement (leave); otherwise candidate 0.
Everything else is unchanged. ORC docstring follows.
"""
"""ORC server (new file, 2026-10-19, multisocket_orc_20261019): copy of policy_server_selrule.py plus the rule
'oracle_endpoint' -- PRIVILEGED selection-ceiling diagnostic. The bridge sends privileged_offset (N, 3) =
[fitting-slot xy - plug-bottom xy (m), gate]; where gate > 0.5 the candidate whose decoded plan displacement over the
select horizon (sum of raw xy actions x 10 = nominal mm, as in multisocket_verifier_train_20261019 'off') lands closest
to the fitting slot is executed, elsewhere candidate 0 (= 'first'). Everything else is unchanged. Original docstring follows.
"""
"""
Local inference server for the manifeel <-> RDP eval bridge.

manifeel (Python 3.8 / torch 2.4 / IsaacGym) and this repo (Python 3.10 /
torch 1.13 / diffusers 0.33) cannot share a single process -- importing
reactive_diffusion_policy from manifeel's conda env fails (missing `timm`,
and even if installed, diffusers 0.11.1 vs 0.33.1 API differs). So instead
this runs the trained LatentDiffusionUnetImagePolicy in its own process
(this repo's rdp_venv) and serves action predictions over a local TCP
socket to a client running inside manifeel's own process
(manifeel/manifeel/eval_rdp_bridge.py).

Wire format (both directions): 4-byte big-endian length prefix + pickle.

Client -> server request (after the one-time handshake), per control step:
    {'obs': {key: np.ndarray [B, n_obs_steps, ...]},
     'extended_obs': {key: np.ndarray [B, original_horizon, ...]}}
Server -> client handshake (sent once, right after connect):
    {'n_obs_steps': int, 'dataset_obs_steps': int,
     'obs_temporal_downsample_ratio': int, 'original_horizon': int,
     'n_action_steps_exec': int,
     'obs_keys': [str, ...], 'extended_obs_keys': [str, ...]}
Server -> client response, per request:
    {'action': np.ndarray [B, n_action_steps_exec, action_dim]}
    or {'error': str}

Usage:
    python policy_server.py \
        --ldp-ckpt /path/to/ldp/checkpoints/latest.ckpt \
        --at-load-dir /path/to/at/checkpoints/latest.ckpt \
        --port 8765
"""
import argparse
import os
import collections
import json
import pathlib
import pickle
import socket
import struct
import sys
import time
import traceback

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from reactive_diffusion_policy.policy.base_image_policy import BaseImagePolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)

REPO_ROOT = pathlib.Path(__file__).resolve().parent


def load_policy(ldp_ckpt: str, at_load_dir: str, task: str, device: str,
                at_cfg: str = 'at_manifeel_pih',
                extended_obs_mode: str = 'padded'):
    # at_cfg must match the AT the LDP was trained against -- it sizes the VAE
    # (n_latent_dims), so loading an l16 checkpoint under the default n=4 config
    # fails with a state_dict shape mismatch.
    payload = torch.load(open(ldp_ckpt, 'rb'), pickle_module=dill)

    # Build from the config the checkpoint was TRAINED with, not a fresh compose
    # of the yaml. Recomposing silently drops every command-line override the
    # training run used, and one of them changes what the network sees:
    # obs_encoder.resize_shape defaults to [240, 320] in the workspace yaml while
    # our runs train with resize_shape=null. Recomposed, a 256x256 observation was
    # stretched to 240x320 and cropped to 216x288 at inference, against 230x230
    # square at training -- the weights load either way, so nothing errors.
    ckpt_cfg = payload.get('cfg')
    if ckpt_cfg is not None:
        cfg = OmegaConf.create(OmegaConf.to_container(ckpt_cfg, resolve=True))
        # The AT is passed explicitly because it lives in a separate checkpoint;
        # everything else comes from the training config.
        cfg.at_load_dir = at_load_dir
        # Imported checkpoints may contain resolved, absolute AT paths rather
        # than interpolations of at_load_dir. Override the actual constructors.
        cfg.policy.at.load_dir = at_load_dir
        if 'at' in cfg.task.dataset:
            cfg.task.dataset.at.load_dir = at_load_dir
        ckpt_task = cfg.task.get('name', '<unknown>')
        if task not in str(ckpt_task):
            print(f"[policy_server] note: --task {task} differs from the checkpoint's "
                  f"task ({ckpt_task}); using the checkpoint's.", file=sys.stderr)
        print(f"[policy_server] config from checkpoint: "
              f"obs_encoder.resize_shape={cfg.policy.obs_encoder.get('resize_shape')} "
              f"random_transforms={cfg.policy.obs_encoder.get('random_transforms')}",
              file=sys.stderr)
    else:
        print("[policy_server] WARNING: checkpoint has no stored cfg; recomposing from "
              "yaml, which may not match how it was trained.", file=sys.stderr)
        with hydra.initialize_config_dir(
                config_dir=str(REPO_ROOT.joinpath('reactive_diffusion_policy', 'config')),
                version_base=None):
            cfg = hydra.compose(
                config_name="train_latent_diffusion_unet_real_image_workspace",
                overrides=[
                    f"task={task}",
                    f"at={at_cfg}",
                    f"at_load_dir={at_load_dir}",
                ])
        OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy: BaseImagePolicy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    # AT is a submodule of the policy but keeps its own normalizer copy;
    # mirrors the same hack eval_real_robot_flexiv.py does.
    policy.at.set_normalizer(policy.normalizer)

    policy.eval().to(torch.device(device))
    policy.num_inference_steps = int(__import__('os').environ.get('RDP_INFER_STEPS', '8'))  # DDIM inference steps (rt_speed_20261113: env override, default 8 as before)

    print(f"[policy_server] loaded policy on {device}: "
          f"n_obs_steps={policy.n_obs_steps} original_horizon={policy.original_horizon} "
          f"action_dim={policy.action_dim}", file=sys.stderr)

    constants = {
        'ldp_checkpoint': str(pathlib.Path(ldp_ckpt).resolve()),
        'at_checkpoint': str(pathlib.Path(at_load_dir).resolve()),
        'inference_steps': policy.num_inference_steps,
        'n_obs_steps': int(policy.n_obs_steps),
        'dataset_obs_steps': int(cfg.task.dataset.n_obs_steps),
        'obs_temporal_downsample_ratio': int(cfg.task.dataset.obs_temporal_downsample_ratio),
        'original_horizon': int(policy.original_horizon),
        'n_action_steps_exec': int(cfg.n_action_steps),
        # which obs/extended_obs keys this particular task config actually
        # needs -- lets the same client work against different task
        # variants (e.g. C8D vs raw tactile_force_field_right) without
        # hardcoding key lists on either side.
        'extended_obs_mode': extended_obs_mode,
        'obs_keys': list(cfg.task.shape_meta.obs.keys()),
        'extended_obs_keys': list(cfg.task.shape_meta.extended_obs.keys()),
    }
    return policy, constants


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b''
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed while reading")
        buf += chunk
    return buf


def recv_msg(conn: socket.socket):
    (length,) = struct.unpack('>I', recv_exact(conn, 4))
    return pickle.loads(recv_exact(conn, length))


def send_msg(conn: socket.socket, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack('>I', len(data)) + data)


def to_torch(np_dict, device):
    return {k: torch.from_numpy(v).to(device=device, dtype=torch.float32)
            for k, v in np_dict.items()}



def obs_features(policy, obs_dict, feat_encoder=None):
    """Verifier / record obs_features (B, 2078). Default: the served policy's own normalizer + obs_encoder (unchanged
    behaviour). feat_encoder (from load_feature_encoder, --verifier-obs-encoder-ckpt) substitutes another LDP's
    obs normalizer + obs_encoder, so a policy trained with its own encoder still feeds the verifier the features it
    was trained on (LDP-S diversity study 2026-10-06)."""
    if feat_encoder is None:
        norm, enc, to = policy.normalizer, policy.obs_encoder, policy.n_obs_steps
    else:
        norm, enc, to = feat_encoder['normalizer'], feat_encoder['obs_encoder'], feat_encoder['n_obs_steps']
    nobs = norm.normalize(obs_dict)
    this = {kk: vv[:, :to].reshape(-1, *vv.shape[2:]) for kk, vv in nobs.items()}
    return enc(this).reshape(next(iter(obs_dict.values())).shape[0], -1)


def load_feature_encoder(ldp_ckpt, device, at_cfg='at_manifeel_pih_l16', at_load_dir=None):
    """Load another LDP checkpoint exactly as load_policy does (EMA model, checkpoint cfg) and keep only its obs
    normalizer + obs_encoder. at_load_dir defaults to the AT path stored in that checkpoint's cfg."""
    if at_load_dir is None:
        ck = torch.load(open(ldp_ckpt, 'rb'), pickle_module=dill, map_location='cpu')
        at_load_dir = OmegaConf.to_container(ck['cfg'], resolve=True)['at_load_dir']
        del ck
    pol, _ = load_policy(ldp_ckpt, at_load_dir, 'feature_encoder', device, at_cfg)
    out = {'obs_encoder': pol.obs_encoder, 'normalizer': pol.normalizer, 'n_obs_steps': int(pol.n_obs_steps),
           'ckpt': str(ldp_ckpt), 'at_load_dir': str(at_load_dir)}
    del pol
    torch.cuda.empty_cache()
    return out


def _decode_plan(policy, cands_c, extended_obs_dict, horizon, ratio):
    """Differentiable open-loop decode of one latent for `horizon` steps (last tactile frame held)."""
    key = next(iter(extended_obs_dict))
    hist = extended_obs_dict[key]
    acts = []
    for _ in range(horizon):
        res = policy.predict_from_latent_action(cands_c, {key: hist}, hist.shape[-2], ratio)
        acts.append(res['action'][:, -1])
        hist = torch.cat([hist, hist[:, -1:]], dim=1)
    return policy.normalizer['action'].normalize(torch.stack(acts, 1))     # (N, H, A)


def _refine_latent(policy, z0, extended_obs_dict, horizon, ratio, verifier, verifier_state,
                   prev_mean, steps, radius):
    """DICE-RL's objective without its machinery: ascend the learned advantage under a proximity
    constraint. Each sample moves along its OWN gradient, so unlike MSE distillation toward a common
    target this cannot average across modes."""
    vm, vst = verifier['model'], verifier['stats']
    zf = lambda x, key: (x - vst[key][0]) / vst[key][1]
    V = zf(verifier_state['obs_features'], 'V'); P = zf(verifier_state['P_generic'], 'P')
    n = z0.shape[0]
    z = z0.clone()
    # cuDNN refuses RNN backward on a module in eval mode; disabling cuDNN for this block keeps the
    # AT decoder's dropout/BN behaviour identical to normal inference (unlike switching to train()).
    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
        for _ in range(steps):
            z = z.detach().requires_grad_(True)
            plan = _decode_plan(policy, z, extended_obs_dict, horizon, ratio)
            pm = plan.mean(1)                                              # (N, A)
            full = pm.norm(dim=-1, keepdim=True)
            dist = (pm - prev_mean).norm(dim=-1, keepdim=True) if prev_mean is not None else torch.zeros_like(full)
            cand = torch.cat([pm, full, dist, torch.ones_like(full)], -1)[:, None]   # (N,1,9)
            A = torch.stack([mm(V, P, torch.zeros((n, 0), device=z.device), torch.zeros(n, device=z.device),
                                zf(cand, 'C'), torch.zeros(n, 1, 2, device=z.device))[3]
                             for mm in verifier.get('members', [vm])]).min(0).values
            g, = torch.autograd.grad(A.sum(), z)
            gn = g / (g.norm(dim=-1, keepdim=True) + 1e-9)
            z = z + (radius / steps) * gn                                  # normalized ascent
            d = z - z0
            dn = d.norm(dim=-1, keepdim=True)
            z = z0 + d * torch.clamp(radius / (dn + 1e-9), max=1.0)        # project onto the trust region
    return z.detach()


def _guide_escape(policy, z0, ext_sub, med, sig, r, horizon, ratio, steps, cap):
    """Normalized ascent of min(||plan(z) - med|| / sig, cap) inside ||z - z0|| <= r ||z0|| (per row)."""
    z = z0.clone(); rad = (r * z0.norm(dim=-1)).unsqueeze(-1)
    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
        for _ in range(steps):
            z = z.detach().requires_grad_(True)
            plan = _decode_plan(policy, z, ext_sub, horizon, ratio).reshape(z.shape[0], -1)
            obj = torch.clamp((plan - med).norm(dim=-1) / sig, max=cap)
            g, = torch.autograd.grad(obj.sum(), z)
            z = z + (rad / steps) * g / (g.norm(dim=-1, keepdim=True) + 1e-9)
            d = z - z0
            z = z0 + d * torch.clamp(rad / (d.norm(dim=-1, keepdim=True) + 1e-9), max=1.0)
    return z.detach()


def cands_on_cuda(obs_dict):
    return torch.cuda.is_available() and next(iter(obs_dict.values())).is_cuda


def _clamp_affine(policy, xgrid):
    """Cached map of the latent-axis constraint into the normalized diffusion space: u, |M^T w|, <w, b>."""
    if '_clamp_aff' not in xgrid:
        M, b = policy.latent_affine()
        w = xgrid['w'].to(M.device, M.dtype)
        wp = M.T @ w
        wn = wp.norm()
        if float(wn) < 1e-8:
            raise RuntimeError('latent_clamp: axis w is orthogonal to the range of the latent map')
        xgrid['_clamp_aff'] = {'u': wp / wn, 'wn': wn, 'wb': (w * b).sum()}
    return xgrid['_clamp_aff']


def _clamp_candidates(policy, obs_dict, draws, noises, g, w, xgrid, info):
    """latent_clamp: for each draw d and grid value g != 0, rerun the sampler from the same x_T with the latent's
    w-component clamped (DDIM x0-projection, policy.make_x0_projection_clamp) to <z0_d, w> + g / mm_per_unit.
    Returns {d: [z for g in grid[1:]]}."""
    aff = _clamp_affine(policy, xgrid)
    sync = cands_on_cuda(obs_dict)
    torch.cuda.synchronize() if sync else None
    t0 = time.time()
    gc = policy.encode_global_cond(obs_dict)                       # (N, Dg), encoded once
    n_env = gc.shape[0]
    jobs = [(d, gi) for d in range(len(draws)) for gi in g[1:]]
    out = []
    chunk = max(int(xgrid.get('clamp_chunk', 16)), 1)
    j = 0
    while j < len(jobs):
        sub = jobs[j:j + chunk]
        noise = torch.cat([noises[d] for d, _ in sub], 0)
        tgt = torch.cat([draws[d] @ w + gi / xgrid['mm_per_unit'] for d, gi in sub], 0)
        c = (tgt - aff['wb']) / aff['wn']
        try:
            z = policy.sample_latent_clamped(gc.repeat(len(sub), 1), noise, aff['u'], c)
        except torch.cuda.OutOfMemoryError:
            if chunk == 1:
                raise
            torch.cuda.empty_cache()
            chunk = max(chunk // 2, 1)
            print(f"[policy_server] clamp tile OOM -> clamp_chunk {chunk}", file=sys.stderr)
            continue
        out.extend(z.view(len(sub), n_env, -1).unbind(0))
        j += len(sub)
    torch.cuda.synchronize() if sync else None
    info['diff_time_clamp_s'] = time.time() - t0
    ng1 = len(g) - 1
    return {d: out[d * ng1:(d + 1) * ng1] for d in range(len(draws))}


def tactile_stall_trigger(extended_obs_dict, gate_base, stuck, thr):
    """xgrid gate 'tactile_stall' (pih_tactile_gate_20261006), sensor-only. contact = max over the refresh's
    extended_obs frames of ||c8d[:3] - episode baseline (mean c8d[:3] of the first 5 requests)|| > thr (thr calibrated
    on the training demos only: free-space EE z > 0.08 p99.9 0.018 < 0.02); trigger = contact AND stuck (server's
    default stall rule). No baseline yet -> contact False. Returns (trigger, contact, indicator) each (N,)."""
    c = extended_obs_dict['contact_8d_right'][..., :3].float()           # (N, T, 3)
    if gate_base is None:
        ind = torch.zeros(c.shape[0], device=c.device)
    else:
        ind = (c - gate_base.to(c.device, c.dtype)[:, None]).norm(dim=-1).amax(1)
    contact = ind > thr
    return contact & stuck.to(contact.device), contact, ind


def gate_allowed_mask(cand_offset_mm, trigger):
    """(k, N) bool: candidate allowed for env. Trigger off -> only grid-offset-0 candidates (the plain draws)."""
    zero = (cand_offset_mm == 0)[:, None]                                 # (k, 1)
    return zero | trigger[None].to(zero.device)


def gate_sign_mask(cand_offset_mm, sign):
    """xgrid gate 'external_sign' (pih_tilt_recovery_20261008/step3): (k, N) bool. sign (N,) from the client:
    0 -> only offset-0 candidates; +1 / -1 -> offset 0 or offsets of that sign; 2 -> all offsets."""
    off = cand_offset_mm[:, None]
    sg = sign.to(off.device)[None]
    return (off == 0) | (sg == 2) | ((sg != 0) & (torch.sign(off) == sg.to(off.dtype)))


QT_FEATURE_SET = 'qt_v1'             # multisocket_qtime_20261020
TERM_SW_MM, TERM_EPS = 15.0, 0.25
STOP_MM = -5.0
MODE_SPACE = {}
HAZ = {}
DEC = {}
GUIDE = dict(rmax=0.5, tau_ref=0.05, steps=4, cap=6.0, force=False, measure_only=False, verifier=None)
VER_S_MIN = 24


def ver_cand_features(pr):
    """pr (k, N, H, A) unnormalized plans -> (k, N, 11) plan-only features (multisocket_critic_offline_20261020 part 2)."""
    xy = pr[..., :2].sum(2) * 10.0; disp = xy.norm(dim=-1)
    return torch.cat([xy, disp[..., None], pr[..., 2:3].sum(2) * 10.0, pr[..., 3:6].sum(2), pr[..., :2].norm(dim=-1).max(2).values[..., None] * 10.0,
                      pr[..., :6, :2].sum(2) * 10.0, (disp >= TERM_SW_MM)[..., None].float()], -1)
DWELL_STAY, DWELL_LEAVE, DWELL_ZPROG_MM = 48, 96, 1.0
VR_FEATURE_SET = 'vretrain_v1'      # pih_verifier_retrain_20261006 feature set (opt-in, --verifier-tactile-offset)
VR_TACTILE_DIM = 26
VR_CAND_DIM = 19
VR_OFFSET_SCALE_MM = 24.0


def vr_tactile_features(tac_hist, tac_base, thr=0.02):
    """Sensor-only tactile summary T (N, 26) for the retrained verifier, from the c8d frames of the last <= 12
    requests (tac_hist: list of (N, 8), oldest first, current request last) and the episode baseline tac_base (N, 8)
    = mean c8d of the first 5 requests: [c8d_now (8), c8d_now - base (8), mean over window of (c8d - base) (8),
    max over window of ||c8d[:3] - base[:3]|| (1), that max > thr (1)]. No privileged input."""
    h = torch.stack(list(tac_hist), 1).float()                            # (N, W, 8)
    b = tac_base.to(h.device, h.dtype)[:, None]
    dev = h - b
    ind = dev[..., :3].norm(dim=-1).amax(1, keepdim=True)                 # (N, 1)
    return torch.cat([h[:, -1], dev[:, -1], dev.mean(1), ind, (ind > thr).to(h.dtype)], 1)


def vr_old_cand_features(plan_mean, lateral, dist, raw_xy):
    """The original verifier's 12 candidate features (k, N, 12): plan mean (6), lateral, dist to recent actions,
    valid flag, raw planned xy unit direction (2), its norm -- identical to the learned_q default branch."""
    xyn = raw_xy.norm(dim=-1, keepdim=True)
    return torch.cat([plan_mean, lateral[..., None], dist[..., None], torch.ones_like(dist[..., None]),
                      raw_xy / (xyn + 1e-9), xyn], -1)


def vr_cand_features(plan_mean, lateral, dist, raw_xy, cand_offset_mm, exec_disp_mm):
    """Retrained-verifier candidate features (k, N, 19) = the original 12 + offset features [offset/24, sign,
    |offset|/24, offset == 0] + executed-window nominal displacement (3, mm / 10). cand_offset_mm (k,) is the
    candidate's own grid value (deployable, chosen by the server); the privileged socket-plug offset never enters."""
    k, n = plan_mean.shape[:2]
    o = cand_offset_mm.to(plan_mean.device, plan_mean.dtype)[:, None, None].expand(k, n, 1)
    off = torch.cat([o / VR_OFFSET_SCALE_MM, torch.sign(o), o.abs() / VR_OFFSET_SCALE_MM, (o == 0).to(o.dtype)], -1)
    return torch.cat([vr_old_cand_features(plan_mean, lateral, dist, raw_xy), off,
                      exec_disp_mm.to(plan_mean.device, plan_mean.dtype) / 10.0], -1)


def explore_offset_choice(cand_offset_mm, Q, eps, rng):
    """explore_q (pih_verifier_retrain_20261006 collection): per env, with prob eps a grid offset value uniformly
    from the grid (incl. 0), else offset 0; then the old verifier's argmax Q among that offset's candidates (one per
    draw). Returns (idx (N,), explored (N,) bool, offset value (N,)). rng: numpy Generator independent of torch."""
    grid = sorted(set(float(v) for v in cand_offset_mm.tolist()), key=lambda v: (abs(v), v))
    n = Q.shape[0]
    explored = rng.random(n) < eps
    pick = rng.integers(0, len(grid), size=n)
    val = np.where(explored, np.asarray(grid)[pick], 0.0).astype(np.float32)
    valt = torch.as_tensor(val, device=Q.device)
    allowed = cand_offset_mm.to(Q.device)[None, :] == valt[:, None]       # (N, k)
    idx = Q.masked_fill(~allowed, float('-inf')).argmax(1)
    return idx, torch.as_tensor(explored, device=Q.device), valt


def draw_noise_seed(req_seed, d):
    """Per-(request seed, draw index) seed for --draw-noise per_draw_seeded (budget-compare study, 2026-10-06).
    req_seed is the bridge's policy_seed + control step, so it is unique per refresh of an episode."""
    return int(np.random.SeedSequence([int(req_seed), int(d), 0xB0D6E7]).generate_state(1, np.uint64)[0]) >> 1


def draw_noise_blocks(n_draws, base, chunk):
    """Diffusion-call layout for per_draw_seeded: draw0 alone, draws 1..base-1 in tiles of `chunk`, then draws
    base..n-1 in tiles of `chunk`. The first `base` draws therefore run in exactly the same batch shapes whatever the
    total budget (Raw8 and Raw40 both: [0] [1-4] [5-7] ...), so they are bitwise identical at the same state."""
    chunk = max(int(chunk), 1)
    cut = min(int(base), n_draws)
    blocks = [[0]] if n_draws > 0 else []
    for lo, hi in ((1, cut), (cut, n_draws)):
        for a in range(max(lo, 1), hi, chunk):
            blocks.append(list(range(a, min(a + chunk, hi))))
    return blocks


def _seeded_draws(policy, obs_dict, extended_obs_dict, ratio, n_draws, draw_noise, chunk, info, cache_cond=False):
    """Independent LDP draws whose initial noise x_T is a function of (request seed, draw index) only: no global RNG
    use, so the draw set does not depend on how many draws are requested or on earlier RNG consumption. The DDIM
    sampler (eta = 0) is deterministic given x_T.
    cache_cond (opt-in --cache-diffusion-cond, 2026-10-08): the diffusion conditioning (normalized obs window ->
    obs_encoder -> global_cond) is computed ONCE for the n_env rows of THIS call and repeated along the batch for every
    draw block (same per-draw x_T, same block layout). It is a local variable: nothing persists across requests.
    extended_obs (tactile) is not used by the latent sampler and is never cached; the decoder gets it fresh."""
    if float(getattr(policy, 'kwargs', {}).get('eta', 0.0)) != 0.0:
        raise NotImplementedError('per_draw_seeded needs the deterministic DDIM sampler (eta = 0)')
    n_env = next(iter(obs_dict.values())).shape[0]
    dev = policy.device
    shape = (n_env, policy.horizon, policy.action_dim)
    noise = []
    for d in range(n_draws):
        gen = torch.Generator(device=dev).manual_seed(draw_noise_seed(draw_noise['seed'], d))
        noise.append(torch.randn(shape, generator=gen, device=dev, dtype=policy.dtype))
    tile = lambda dct, m: {kk: vv.repeat(m, *([1] * (vv.dim() - 1))) for kk, vv in dct.items()}
    parts = [None] * n_draws
    todo = draw_noise_blocks(n_draws, draw_noise.get('base', 8), chunk)
    info['draw_calls'] = 0
    info['draw_oom'] = 0
    gc = None
    if cache_cond:
        gc = policy.encode_global_cond(obs_dict)          # (n_env, Dg), once per request
        info['cond_encodes'] = 1
    while todo:
        blk = todo.pop(0)
        m = len(blk)
        try:
            if cache_cond:
                out = policy.sample_latent_from_global_cond(gc.repeat(m, 1) if m > 1 else gc,
                                                            torch.cat([noise[d] for d in blk], 0))
            else:
                out = policy.predict_action(tile(obs_dict, m) if m > 1 else obs_dict, dataset_obs_temporal_downsample_ratio=ratio,
                                            extended_obs_dict=tile(extended_obs_dict, m) if m > 1 else extended_obs_dict,
                                            return_latent_action=True,
                                            init_noise=torch.cat([noise[d] for d in blk], 0))['action'][:, 0, :]
        except torch.cuda.OutOfMemoryError:
            # exception path only: split the tile (noise per draw unchanged; batch shape differs -> flagged)
            if m == 1:
                raise
            torch.cuda.empty_cache()
            info['draw_oom'] += 1
            print(f"[policy_server] seeded draw tile OOM -> split {m}", file=sys.stderr)
            todo = [blk[:m // 2], blk[m // 2:]] + todo
            continue
        info['draw_calls'] += 1
        for j, d in enumerate(blk):
            parts[d] = out[j * n_env:(j + 1) * n_env]
    return parts


def _tsync(on, obs_dict):
    if on and cands_on_cuda(obs_dict):
        torch.cuda.synchronize()
    return time.perf_counter()


def load_dec(dec_dir, curve, eps, device):
    import pickle
    d = pathlib.Path(dec_dir); m = np.load(d / 'dec_model.npz')
    DEC.update({k: m[k] for k in m.files})
    nets = []
    for q in range(3):
        net = torch.nn.Sequential(torch.nn.Linear(146, 512), torch.nn.SiLU(), torch.nn.Linear(512, 512), torch.nn.SiLU(), torch.nn.Linear(512, 69))
        net.load_state_dict(torch.load(d / f'dyn_{q}.pt', map_location='cpu')); net.eval(); nets.append(net.to(device))
    DEC['nets'] = nets; DEC['device'] = device
    with open(d / 'v_obs.pkl', 'rb') as fh:
        DEC['V'] = pickle.load(fh)
    hc = json.loads(pathlib.Path(curve).read_text()); DEC.update(a=float(hc['a']), b=float(hc['b']), d_cap=int(hc['d_cap']), eps=float(eps), lmax=1000.0)
    print('[policy_server] DEC', {k: DEC[k] for k in ('a', 'b', 'd_cap', 'eps', 'lmax')}, file=sys.stderr)


def dec_plan_feats(pr, c):
    """pr (H, 6) plan_raw of one candidate, c (3,) -> 77 features (as multisocket_decomp_verifier_offline_20261022.plan_feats)"""
    p = 0.01 * pr[:, :3].sum(0); pn = np.linalg.norm(p); cn = np.linalg.norm(c)
    cos = float(p @ c / (pn * cn)) if pn > 0 and cn > 0 else 0.0
    return np.concatenate([pr[:, :6].ravel(), p, [pn, cos]]).astype(np.float32)


def dec_score(feats, c, d, pr):
    """A_hat (k, N) = V(x + f(x, u_k)) - V(x); feats (N, 2078) fp16-rounded obs features, c (N, 3), d (N,), pr (k, N, H, 6)"""
    k, n = pr.shape[:2]
    z = (feats - DEC['pca_mean']) @ DEC['pca_components'].T
    xs = (np.hstack([z, c, np.linalg.norm(c, axis=1, keepdims=True), d[:, None].astype(np.float64)]) - DEC['x_mu']) / DEC['x_sd']
    U = np.stack([np.stack([dec_plan_feats(pr[kk, i], c[i]) for i in range(n)]) for kk in range(k)])         # (k, N, 77)
    inp = np.concatenate([np.repeat(xs[None], k, 0), (U - DEC['u_mu']) / DEC['u_sd']], -1).reshape(k * n, -1).astype(np.float32)
    with torch.no_grad():
        zt = torch.as_tensor(inp, device=DEC['device']); dl = np.mean([net(zt).cpu().numpy() for net in DEC['nets']], 0)
    x1 = np.repeat(xs[None], k, 0).reshape(k * n, -1) + dl
    V0 = DEC['V'].predict_proba(xs[:, :64])[:, 1]; V1 = DEC['V'].predict_proba(x1[:, :64])[:, 1].reshape(k, n)
    return V1 - V0[None]


COH = {'lam': 0.0, 'm': 4, 'd': 0, 'R': 12, 'prev': None}
PREGEN = {'c': None, 'used': 0, 'made': 0}


def select_latent(policy, constants, obs_dict, extended_obs_dict, rule, k, horizon, recent_actions,
                  z_history=None, stuck_drop=0.002, stuck_window=24, rng=None, record=False, privileged_offset=None, qt_extra=None, switch_flag=None,
                  verifier=None, verifier_state=None, residual=None, refine_steps=5, refine_radius=0.25,
                  select_every=1, refresh_idx=0, select_pool=0, head=None, local_radius=0.0,
                  draw_chunk=4, codebook=None, local_keep=0, local_cov=None, given_cands=None, xgrid=None,
                  gate_base=None, explore=None, record_vr=False, draw_noise=None, timing=False, cache_cond=False,
                  ext_sign=None, switch_tau=None, term_x=None, step_now=None, stop_flag=None, mode_mu=None, mode_gate=None, mode_s=None, soft_seed=None, req_seed=0, rim_orc=None, explore_k=None, stall_d=None, dec_feats=None, dec_x=None):
    """Per-refresh selection among k diffusion samples drawn from the same observation.

    Candidate 0 is exactly the draw normal RDP would use (same RNG state), so 'first' reproduces
    normal RDP with the same sampling budget. Each candidate is decoded open-loop for `horizon`
    steps (last real tactile frame held) and summarized in normalized action units.
      consistent / inconsistent : plan mean closest / farthest from the last `horizon` executed actions
      random                    : uniform among the k candidates (same distribution as 'first')
      max_magnitude             : largest mean per-step lateral (xy) action norm
      inconsistent_when_stuck   : 'inconsistent' only for envs whose end-effector z dropped less than
                                  `stuck_drop` m over the last `stuck_window` steps, else candidate 0
      max_magnitude_when_stuck  : 'max_magnitude' only when stuck, else candidate 0
      learned_q / learned_vq    : argmax of the learned dueling Q over candidates (verifier_model.Verifier);
                                  learned_vq only where sigmoid(V) < tau, else candidate 0. Needs >= 24 steps
                                  of history; earlier refreshes use candidate 0.
      oracle_direction          : like oracle_direction_when_stuck but at every refresh (privileged, diagnostic only)
      oracle_direction_when_stuck / anti_direction_when_stuck : when stuck, the candidate whose raw planned
                                  xy motion is most / least aligned (cosine) with the privileged socket-plug xy
                                  offset sent by the client (diagnostic only), else candidate 0
      xgrid (opt-in, 2026-10-05 structured-AT study): candidates = draw0 shifted by each value of a fixed grid
                                  of executed-window x offsets (nominal mm = sum of action x over the executed
                                  window x 10), followed by draws 1..D-1 unshifted. mode 'latent_axis': z0 + (g /
                                  mm_per_unit) * w (post-hoc axis or AT-S structured direction); mode 'action':
                                  latent z0 unchanged, g/(10*exec_steps) added to every decoded action's x.
                                  Grid value 0 must come first so candidate 0 stays the normal RDP draw.
      draw_noise (opt-in, budget-compare study 2026-10-06): {'mode': 'per_draw_seeded', 'seed': request seed,
                                  'base': B}; draw d's x_T comes from its own generator seeded by (seed, d), and the
                                  first B draws run in budget-independent batch shapes (draw_noise_blocks), so draws
                                  0..B-1 are bitwise the same for any number of draws >= B. info['base_draws'].
      timing (opt-in): synchronized perf_counter marks info['t_mark_*'] (start, after diffusion, after decode).
      cache_cond (opt-in, per_draw_seeded only): encode the diffusion conditioning once per request (_seeded_draws).
    Returns (latent, chosen_idx, info) where info holds per-refresh diagnostics.
    """
    t_start = _tsync(timing, obs_dict)
    ratio = constants['obs_temporal_downsample_ratio']
    info_t = {'t_mark_start': t_start} if timing else {}
    SELECTING = ('consistent', 'inconsistent', 'random', 'max_magnitude', 'inconsistent_when_stuck',
                 'max_magnitude_when_stuck', 'oracle_direction_when_stuck', 'anti_direction_when_stuck', 'oracle_direction',
                 'oracle_phase', 'learned_q', 'learned_vq', 'oracle_land', 'explore_q', 'oracle_endpoint', 'dwell_rule', 'rand_switch', 'term_switch', 'term_guide', 'stop_first', 'mode_first', 'soft_tilt', 'soft_rimorc', 'soft_explore', 'haz', 'haz_soft', 'dec_soft')
    skip = rule in SELECTING and select_every > 1 and (refresh_idx % select_every) != 0
    if rule in ('residual', 'refine', 'perturb') or skip:
        k = 1
    info = dict(info_t)
    n_draw = max(k, select_pool) if (select_pool > k and not skip and rule not in ('residual', 'refine', 'perturb')) else k
    # Keep an LDP prefix when mixing. Candidate 0 retains the baseline draw.
    use_head = head is not None and verifier_state is not None
    use_local = local_radius > 0 and verifier_state is not None and not skip
    if local_radius < 0 or (local_radius > 0 and (head is not None or select_pool > 0 or rule != 'learned_q' or (k % 2 and not skip and local_keep == 0))):
        raise ValueError('Local mixing requires learned_q, even K, no head/pool, nonnegative radius')
    keep = int(head.get('keep_ldp', 0)) if use_head else n_draw
    use_xgrid = xgrid is not None and not skip and given_cands is None
    if use_xgrid:
        if use_head or use_local or codebook is not None or select_pool > 0:
            raise ValueError('xgrid cannot be combined with head/local/codebook/pool')
        n_cand_x = len(xgrid['mm']) * xgrid['draws'] if xgrid.get('all_draws') else len(xgrid['mm']) + xgrid['draws'] - 1
        if n_cand_x != k or xgrid['mm'][0] != 0:
            raise ValueError('xgrid: candidate count (grid x draws with --xgrid-all-draws, else grid + draws - 1) '
                             'must equal select_k and grid[0] must be 0')
        keep = n_draw = int(xgrid['draws'])
        if xgrid['mode'] == 'latent_clamp' and not xgrid.get('all_draws'):
            raise ValueError('xgrid latent_clamp requires --xgrid-all-draws')
    use_clamp = use_xgrid and xgrid['mode'] == 'latent_clamp'
    noises = [] if use_clamp else None       # latent_clamp: initial x_T of every draw (captured, no extra RNG use)
    if codebook is not None:
        # Coarse-mode exploration: one diffusion draw, then every code with that draw's own
        # residual, z' = c + (z0 - nearest_code(z0)). Measured on this LDP's latents the eight
        # codes span 92.9 deg of 8-step heading against 14.8 for eight samples, at 1/8 the draws.
        keep = 1
    if use_head and not 0 <= keep < n_draw:
        raise ValueError('head keep_ldp must be smaller than the candidate budget')
    if use_local:
        # local_all: one diffusion draw, every other candidate a perturbation of it. At r=0.1 the
        # perturbations sit at the policy's own fine-variation scale (0.67 vs 0.71 for fresh draws)
        # and the verifier picked them 70% of the time, so this is the 1/8-cost version.
        keep = int(local_keep) if local_keep > 0 else n_draw // 2
        if not 1 <= keep < n_draw:
            raise ValueError('local_keep must be in [1, k-1]')
    if given_cands is not None:
        # Two-stage selection hands the scorer a ready-made candidate set (stage-1 winner plus
        # perturbations of it); nothing is drawn here.
        cands = given_cands
        info['candidate_source'] = torch.tensor([0] + [2] * (cands.shape[0] - 1), device=cands.device)
    else:
        parts = []
        nk = {'noise_out': noises} if use_clamp else {}
        if draw_noise is not None and (use_clamp or use_head or use_local or codebook is not None or n_draw != keep):
            raise ValueError('draw_noise per_draw_seeded supports plain draws and latent_axis/action xgrid only')
        if cache_cond and draw_noise is None:
            raise ValueError('cache_cond (--cache-diffusion-cond) is implemented for draw_noise per_draw_seeded only')
        if use_clamp:
            torch.cuda.synchronize() if cands_on_cuda(obs_dict) else None
            t_draw0 = time.time()
        # Candidate 0 is drawn alone so it stays the exact draw normal RDP would use (same RNG
        # state). The remaining draws are tiled along the batch and run `draw_chunk` at a time:
        # at 50 envs this turns 8 sequential diffusion calls per refresh into 3, which was most
        # of the refresh wall-clock. --draw-chunk 1 reproduces the sequential loop exactly.
        if draw_noise is not None:
            parts = _seeded_draws(policy, obs_dict, extended_obs_dict, ratio, keep, draw_noise, draw_chunk, info,
                                  cache_cond=cache_cond)
            info['base_draws'] = torch.stack(parts[:min(int(draw_noise.get('base', 8)), keep)])
            rest = 0
        elif __import__('os').environ.get('RDP_DRAW_SINGLE', '0') == '1' and not use_clamp:
            # rt_speed2_20261113: all candidates in ONE generator call (candidate 0 is no longer the lone normal-RDP draw; same distribution)
            _ne = next(iter(obs_dict.values())).shape[0]; _tl = lambda d, m: {kk: vv.repeat(m, *([1] * (vv.dim() - 1))) for kk, vv in d.items()}
            # rt_final AMEND1 (2026-11-14): at 25 envs the 400-row call ran out of GPU memory next to the simulator. The tiled batch is split
            # into calls of at most RDP_DRAW_SINGLE_ROWS rows (default 128 = the size the 8-env smoke passed with). One env (real robot) is still one call.
            _mr = max(1, int(__import__('os').environ.get('RDP_DRAW_SINGLE_ROWS', '128')) // _ne); _k = 0
            while _k < keep:
                _m = min(_mr, keep - _k)
                _o = policy.predict_action(_tl(obs_dict, _m), dataset_obs_temporal_downsample_ratio=ratio, extended_obs_dict=_tl(extended_obs_dict, _m),
                                           return_latent_action=True)['action'][:, 0, :]
                parts.extend(_o.view(_m, _ne, -1).unbind(0)); _k += _m
            rest = 0
        else:
            parts.append(policy.predict_action(obs_dict, dataset_obs_temporal_downsample_ratio=ratio,
                                               extended_obs_dict=extended_obs_dict,
                                               return_latent_action=True, **nk)['action'][:, 0, :])
            rest = keep - 1
        n_env = parts[0].shape[0]
        chunk = max(int(draw_chunk), 1)
        tile = lambda d, m: {kk: vv.repeat(m, *([1] * (vv.dim() - 1))) for kk, vv in d.items()}
        while rest > 0:
            m = min(chunk, rest)
            n_noise = len(noises) if use_clamp else 0
            try:
                out = policy.predict_action(tile(obs_dict, m), dataset_obs_temporal_downsample_ratio=ratio,
                                            extended_obs_dict=tile(extended_obs_dict, m),
                                            return_latent_action=True, **nk)['action'][:, 0, :]
            except torch.cuda.OutOfMemoryError:
                if use_clamp:
                    del noises[n_noise:]
                # The sim shares the card; shrink the tile instead of dying mid-rollout.
                if chunk == 1:
                    raise
                torch.cuda.empty_cache()
                chunk = max(chunk // 2, 1)
                print(f"[policy_server] draw tile OOM -> draw_chunk {chunk}", file=sys.stderr)
                continue
            parts.extend(out.view(m, n_env, -1).unbind(0))
            rest -= m
        if use_clamp:
            # per-draw initial noise, same order as parts: draw0 alone, then each tile is m blocks of n_env rows
            noises = [nz for blk in noises for nz in blk.view(-1, n_env, *blk.shape[1:]).unbind(0)]
            assert len(noises) == keep
            torch.cuda.synchronize() if cands_on_cuda(obs_dict) else None
            info['diff_time_draws_s'] = time.time() - t_draw0
        if timing:
            info['t_mark_gen'] = _tsync(True, obs_dict)
        if use_head:
            hm, hst = head['model'], head['stats']
            head_input = verifier_state['obs_features']
            if head.get('conditioning', 'vision') == 'vision_pose_history':
                head_input = torch.cat([head_input, verifier_state['P_generic']], -1)
            v = (head_input - hst['V'][0]) / hst['V'][1]
            n_ = v.shape[0]
            zz = hm.sample(v.repeat(n_draw - keep, 1)).view(n_draw - keep, n_, -1)
            generated = zz * hst['Z'][1] + hst['Z'][0]
            parts.extend(generated.unbind(0))
        if use_local:
            # anchors cycle over the kept draws so every draw gets perturbations
            anchors = torch.stack([parts[i % keep] for i in range(n_draw - keep)]) if local_keep > 0 else torch.stack(parts)
            if local_cov is not None:
                # Shaped perturbation: noise only along the within-state principal directions, with the
                # policy's own per-direction spread. Records show within-state variation is low-rank
                # (16 of 128 PCs carry 71%); isotropic noise puts ~88% of its energy where the policy
                # never varies. local_radius is then a multiplier on the natural scale (1 = natural).
                Lm = local_cov                                                      # (D, m) = V_w sqrt(lambda)
                eps = torch.randn(*anchors.shape[:-1], Lm.shape[1], device=anchors.device)
                modified = anchors + local_radius * (eps @ Lm.T)
            else:
                direction = torch.randn_like(anchors)
                direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                modified = anchors + local_radius * anchors.norm(dim=-1, keepdim=True) * direction
            parts.extend(modified.unbind(0))
        if codebook is not None:
            z0 = parts[0]                                                  # (N, D)
            dd = ((z0[:, None, :] - codebook[None]) ** 2).sum(-1)          # (N, k_codes)
            resid = z0 - codebook[dd.argmin(1)]                            # (N, D)
            parts = [z0] + [codebook[j][None] + resid for j in range(codebook.shape[0])]
            n_draw = len(parts)
        if use_xgrid:
            # draw0 x grid, then the remaining independent draws unshifted
            z0 = parts[0]
            g = torch.as_tensor(xgrid['mm'], device=z0.device, dtype=z0.dtype)            # (G,)
            ng = g.shape[0]
            n_env_ = z0.shape[0]
            act_off = torch.zeros(ng + keep - 1, n_env_, xgrid['action_dim'], device=z0.device, dtype=z0.dtype)
            if xgrid['mode'] in ('latent_axis', 'latent_clamp'):
                w = xgrid['w'].to(z0.device, z0.dtype)
                shifted = [z0 + (gi / xgrid['mm_per_unit']) * w[None] for gi in g]
            else:
                shifted = [z0.clone() for _ in range(ng)]
                act_off[:ng, :, 0] = (g / (10.0 * xgrid['exec_steps']))[:, None]
            if xgrid.get('all_draws'):
                # every independent draw x the grid (draw-major): a superset of the plain draws (grid[0] = 0), so
                # the non-x diversity of independent samples is kept (2026-10-06 fix: draw0-only grid dropped it)
                act_off = torch.zeros(ng * keep, n_env_, xgrid['action_dim'], device=z0.device, dtype=z0.dtype)
                parts_x = []
                if use_clamp:
                    clamped = _clamp_candidates(policy, obs_dict, parts[:keep], noises, g, w, xgrid, info)
                for d in range(keep):
                    zd = parts[d]
                    if xgrid['mode'] == 'latent_axis':
                        parts_x += [zd + (gi / xgrid['mm_per_unit']) * w[None] for gi in g]
                    elif use_clamp:
                        parts_x += [zd] + clamped[d]          # g = 0 is the plain draw itself
                    else:
                        parts_x += [zd.clone() for _ in range(ng)]
                        act_off[d * ng:(d + 1) * ng, :, 0] = (g / (10.0 * xgrid['exec_steps']))[:, None]
                parts = parts_x
                if use_clamp:
                    cz = torch.stack(parts)                                           # (k, N, D)
                    base = torch.stack([parts[d * ng] for d in range(keep)]).repeat_interleave(ng, 0)
                    dz = cz - base
                    along = dz @ w
                    info['clamp_achieved'] = cz @ w                                   # (k, N)
                    info['clamp_target'] = (base @ w) + (g / xgrid['mm_per_unit']).repeat(keep)[:, None]
                    info['cand_offaxis'] = (dz - along[..., None] * w).norm(dim=-1)   # 0 for P1x by construction
                    info['cand_dz_norm'] = dz.norm(dim=-1)
                info['act_offset'] = act_off
                info['cand_offset_mm'] = g.repeat(keep)
                info['candidate_source'] = torch.tensor(([0] + [4] * (ng - 1)) + ([1] + [4] * (ng - 1)) * (keep - 1), device=z0.device)
            else:
                parts = shifted + parts[1:]
                info['act_offset'] = act_off
                info['cand_offset_mm'] = torch.cat([g, torch.zeros(keep - 1, device=g.device, dtype=g.dtype)])
                info['candidate_source'] = torch.tensor([0] + [4] * (ng - 1) + [1] * (keep - 1), device=z0.device)
            n_draw = keep = len(parts)
        cands = torch.stack(parts)
        if not use_xgrid:
          info['candidate_source'] = torch.tensor([0] * keep + [3 if codebook is not None else (2 if use_local else 1)] * (n_draw - keep), device=cands.device)
        if n_draw > k and codebook is None:
            # Down-select k mutually distant latents BEFORE decoding, so the decode/score budget is
            # unchanged while the effective diversity of the candidate set goes up. Candidate 0 is always
            # kept so it still equals the latent normal RDP would use.
            pool, N_, _ = cands.shape
            ar_ = torch.arange(N_, device=cands.device)
            taken = torch.zeros(pool, N_, dtype=torch.bool, device=cands.device)
            sel = torch.zeros(k, N_, dtype=torch.long, device=cands.device)
            taken[0] = True
            cur = cands[0]
            mind = torch.full((pool, N_), float('inf'), device=cands.device)
            for j in range(1, k):
                mind = torch.minimum(mind, (cands - cur[None]).norm(dim=-1))
                nxt = mind.masked_fill(taken, -1.0).argmax(0)          # (N,)
                sel[j] = nxt
                taken[nxt, ar_] = True
                cur = cands[nxt, ar_]
            cands = cands[sel, ar_[None]]                              # (k, N, D)
            info['pool'] = pool
            info['candidate_source'] = info['candidate_source'][sel]
    k_, n = cands.shape[:2]
    k = k_                                                             # codebook may change the count
    ar = torch.arange(n, device=cands.device)
    need_plans = not skip and rule not in ('residual', 'refine', 'perturb') and (record or rule not in ('first', 'random') or verifier is not None)
    raw_xy = None
    have_hist = len(recent_actions) >= horizon
    if need_plans:
        key = next(iter(extended_obs_dict))
        # All k candidates go through the decoder as one batch of k*N rows. The decoder is
        # per-sample, so this equals the former per-candidate loop (checked to float
        # precision) with k times fewer calls -- 64 -> 8 per refresh at k=8, H=8.
        z_all = cands.reshape(k * n, -1)
        hist = extended_obs_dict[key].repeat(k, *([1] * (extended_obs_dict[key].dim() - 1)))
        acts = []
        # xgrid also decodes the whole executed window (exec_steps >= horizon) for the per-candidate records;
        # the decode is causal step by step, so the first `horizon` steps are the same as before.
        n_dec = max(horizon, int(xgrid['exec_steps'])) if use_xgrid else horizon
        try:
            for _ in range(n_dec):
                res = policy.predict_from_latent_action(z_all, {key: hist}, hist.shape[-2], ratio)
                acts.append(res['action'][:, -1])
                hist = torch.cat([hist, hist[:, -1:]], dim=1)
            raw_all = torch.stack(acts, 1)                       # (k*N, n_dec, A)
        except torch.cuda.OutOfMemoryError:
            # exception path only (2026-10-06 verifier retrain, 72 candidates x 50 envs share the card with the sim):
            # the decoder is per-sample, so decoding the rows in chunks gives the same plans
            acts = res = None
            torch.cuda.empty_cache()
            print('[policy_server] decode OOM -> chunked decode', file=sys.stderr)
            hist0 = extended_obs_dict[key].repeat(k, *([1] * (extended_obs_dict[key].dim() - 1)))
            outs_ = []
            step_ = max(n, (k * n) // 4)
            for r0 in range(0, k * n, step_):
                h_ = hist0[r0:r0 + step_]
                a_ = []
                for _ in range(n_dec):
                    res = policy.predict_from_latent_action(z_all[r0:r0 + step_], {key: h_}, h_.shape[-2], ratio)
                    a_.append(res['action'][:, -1])
                    h_ = torch.cat([h_, h_[:, -1:]], dim=1)
                outs_.append(torch.stack(a_, 1))
            raw_all = torch.cat(outs_, 0)
        if use_xgrid:
            raw_all = raw_all + info['act_offset'].reshape(k * n, 1, -1)
            ex = raw_all[:, :int(xgrid['exec_steps']), :3].sum(1).view(k, n, 3) * 10.0
            info['cand_exec_disp_mm'] = ex                       # nominal mm over the executed window
            if use_clamp:
                info['cand_exec_sum_full'] = raw_all[:, :int(xgrid['exec_steps'])].sum(1).view(k, n, -1)   # all action dims
            raw_all = raw_all[:, :horizon]
        info['plan_raw'] = raw_all.view(k, n, horizon, -1)      # unnormalized (k, N, H, A); recorded only with RECORD_PLANS
        plans = policy.normalizer['action'].normalize(raw_all).view(k, n, horizon, -1)   # (k, N, H, A)
        raw_xy = raw_all[..., :2].mean(1).view(k, n, 2)          # (k, N, 2)
        summ = plans.mean(2)                                     # (k, N, A)
        lateral = plans[..., :2].norm(dim=-1).mean(2)            # (k, N)
        info.update(plan_mean=summ, lateral=lateral, plan_full=plans)
        if have_hist:
            prev = policy.normalizer['action'].normalize(torch.stack(list(recent_actions), 1)).mean(1)
            info['dist'] = (summ - prev[None]).norm(dim=-1)      # (k, N)
            info['recent_mean'] = prev
    if timing:
        info.setdefault('t_mark_gen', info['t_mark_start'])
        info['t_mark_dec'] = _tsync(True, obs_dict)
    stuck = torch.zeros(n, dtype=torch.bool, device=cands.device)
    if z_history is not None and len(z_history) > stuck_window:
        zs = list(z_history)
        stuck = (zs[-1 - stuck_window] - zs[-1]) < stuck_drop
    info['stuck'] = stuck
    use_gate = use_xgrid and xgrid.get('gate', 'none') == 'tactile_stall'
    if use_gate:
        info['gate_trigger'], info['gate_contact'], info['gate_contact_ind'] = tactile_stall_trigger(
            extended_obs_dict, gate_base, stuck, float(xgrid['gate_thr']))
    use_sign = use_xgrid and xgrid.get('gate', 'none') == 'external_sign'
    if use_sign:
        if ext_sign is None:
            raise RuntimeError("xgrid gate external_sign needs 'xgrid_sign' (N,) in every request")
        info['ext_sign'] = torch.as_tensor(np.asarray(ext_sign), device=cands.device, dtype=torch.long)
    if skip:
        info['cands'] = cands
        return cands[0], torch.zeros(n, dtype=torch.long, device=cands.device), info
    if privileged_offset is not None and raw_xy is not None:
        po = torch.as_tensor(privileged_offset, device=cands.device, dtype=raw_xy.dtype)    # (N, 2) or (N, 5) = offset + tilt axis
        off = po[:, :2]
        if po.shape[1] >= 5:
            info['tilt_axis'] = po[:, 2:5]          # privileged: unit axis rotating the plug upright when tilted & in contact, else 0
        info['align'] = (raw_xy * off[None]).sum(-1) / (raw_xy.norm(dim=-1) * off.norm(dim=-1)[None] + 1e-9)  # (k, N)
    idx = torch.zeros(n, dtype=torch.long, device=cands.device)
    if rule == 'perturb':
        # control for 'refine': same displacement magnitude, random direction. If this is as harmful as
        # the value gradient, the damage comes from leaving the sample manifold, not from the critic.
        info['cands'] = cands
        if verifier_state is None:
            info['refine_move'] = torch.zeros(n, device=cands.device)
            return cands[0], idx, info
        g = torch.randn_like(cands[0])
        zr = cands[0] + refine_radius * g / (g.norm(dim=-1, keepdim=True) + 1e-9)
        info['refine_move'] = (zr - cands[0]).norm(dim=-1)
        return zr, idx, info
    if rule == 'refine':
        info['cands'] = cands
        if verifier is None or verifier_state is None:
            info['refine_move'] = torch.zeros(n, device=cands.device)
            return cands[0], idx, info
        prev = (policy.normalizer['action'].normalize(torch.stack(list(recent_actions), 1)).mean(1)
                if have_hist else None)
        zr = _refine_latent(policy, cands[0].detach(), extended_obs_dict, horizon, ratio,
                            verifier, verifier_state, prev, refine_steps, refine_radius)
        info['refine_move'] = (zr - cands[0]).norm(dim=-1)
        return zr, idx, info
    if rule == 'residual':
        info['cands'] = cands
        if residual is None or verifier_state is None:
            info['residual_norm'] = torch.zeros(n, device=cands.device)   # before step 36: plain RDP draw
            return cands[0], idx, info
        rm, rst = residual['model'], residual['stats']
        z = lambda x, key: (x - rst[key][0]) / rst[key][1]
        zs = z(cands[0], 'Z')
        dz = rm(z(verifier_state['obs_features'], 'V'), z(verifier_state['P_generic'], 'P'), zs)
        info['residual_norm'] = dz.norm(dim=-1)
        return (zs + dz) * rst['Z'][1] + rst['Z'][0], idx, info
    if rule == 'random':
        idx = torch.as_tensor(rng.integers(0, k, size=n), device=cands.device)
    elif rule == 'max_magnitude':
        idx = info['lateral'].argmax(0)
    elif rule == 'max_magnitude_when_stuck':
        idx = torch.where(stuck, info['lateral'].argmax(0), idx)
    elif rule in ('learned_q', 'learned_vq', 'explore_q'):
        if have_hist and verifier_state is not None:
            vm, vst = verifier['model'], verifier['stats']
            z = lambda x, key: (x - vst[key][0]) / vst[key][1]
            dist = info['dist']                                      # (k, N)
            generic = bool(getattr(vm, 'generic', False))
            vr = getattr(vm, 'feature_set', None) == VR_FEATURE_SET
            T_in = torch.zeros((n, 0), device=cands.device)
            if getattr(vm, 'feature_set', None) == QT_FEATURE_SET:
                if 'QT' not in verifier_state or 'plan_raw' not in info:
                    raise RuntimeError('qt_v1 verifier needs the bridge --send_qt_extra and decoded plans (--record-plans)')
                from vms_features_20261019 import cand_features as vms_cand_features
                cand = vms_cand_features(info['plan_mean'], info['lateral'], dist, info['plan_raw'].float()).permute(1, 0, 2)  # (N,k,13)
                qe = verifier_state['QT']
                state_flag = qe[:, 53]
                P_in = qe[:, :37]
                T_in = z(qe[:, 37:53], 'T') if vm.dt > 0 else torch.zeros((n, 0), device=cands.device)
                verifier_state['obs_features'] = verifier_state['obs_features'].half().float()   # = recorded training features
            elif vr:   # retrained verifier (opt-in): original 12 + offset features, tactile T, P, stuck flag
                if 'cand_offset_mm' not in info or 'T' not in verifier_state:
                    raise RuntimeError('vretrain_v1 verifier needs --xgrid-mode and tactile history')
                cand = vr_cand_features(info['plan_mean'], info['lateral'], dist, raw_xy, info['cand_offset_mm'],
                                        info['cand_exec_disp_mm']).permute(1, 0, 2)                 # (N,k,19)
                state_flag = stuck.float()
                P_in = verifier_state['P']
                T_in = z(verifier_state['T'], 'T')
            elif bool(getattr(vm, 'rawplan', False)):   # no hand-designed candidate features: the decoded plan itself
                k_, n_, H_, A_ = info['plan_full'].shape
                cand = torch.cat([info['plan_full'].reshape(k_, n_, H_ * A_), dist[..., None],
                                  torch.ones_like(dist[..., None])], -1).permute(1, 0, 2)          # (N,k,H*A+2)
                state_flag = torch.zeros(n, device=cands.device)
                P_in = verifier_state['P_generic']
            elif generic:   # task-agnostic: action-space magnitude only, no axis singled out, no stuck flag
                full = info['plan_mean'].norm(dim=-1, keepdim=True)
                cand = torch.cat([info['plan_mean'], full, dist[..., None],
                                  torch.ones_like(dist[..., None])], -1).permute(1, 0, 2)          # (N,k,9)
                state_flag = torch.zeros(n, device=cands.device)
                P_in = verifier_state['P_generic']
            else:
                xyn = raw_xy.norm(dim=-1, keepdim=True)              # (k, N, 1)
                cand = torch.cat([info['plan_mean'], info['lateral'][..., None], dist[..., None],
                                  torch.ones_like(dist[..., None]), raw_xy / (xyn + 1e-9), xyn], -1).permute(1, 0, 2)  # (N,k,12)
                state_flag = stuck.float()
                P_in = verifier_state['P']
            zv, zp2, zc = z(verifier_state['obs_features'], 'V'), z(P_in, 'P'), z(cand, 'C')
            rxy = raw_xy.permute(1, 0, 2) if raw_xy is not None else torch.zeros(n, cand.shape[1], 2, device=cands.device)
            outs = [mm(zv, zp2, T_in, state_flag, zc, rxy)
                    for mm in verifier.get('members', [vm])]
            v, dd = outs[0][0], outs[0][1]
            A_all = torch.stack([o[3] for o in outs])               # qconf_20261117: (M, N, k)
            A = A_all.min(0).values     # pessimistic across the ensemble
            Q = v[:, None] + A - A.mean(1, keepdim=True)
            if os.environ.get('RDP_QZERO') == '1':   # qlow_rule_20261116: no Q -- first allowed candidate (Q = -1000 * index); belief rule masks unchanged
                Q = -1000.0 * torch.arange(Q.shape[1], device=Q.device, dtype=Q.dtype)[None].expand(Q.shape[0], -1).clone()
            if use_gate:   # trigger off -> argmax restricted to the offset-0 candidates; Q itself is recorded unmasked
                allowed = gate_allowed_mask(info['cand_offset_mm'], info['gate_trigger']).T   # (N, k)
                best = Q.masked_fill(~allowed, float('-inf')).argmax(1)
            elif use_sign:   # sign 0 -> offset-0 candidates only; Q recorded unmasked
                allowed = gate_sign_mask(info['cand_offset_mm'], info['ext_sign']).T                # (N, k)
                best = Q.masked_fill(~allowed, float('-inf')).argmax(1)
            else:
                Qs = Q
                if COH['lam'] > 0 and COH['prev'] is not None and 'plan_raw' in info and COH['prev'].shape[0] == Q.shape[0]:
                    d_, m_, R_ = COH['d'], COH['m'], COH['R']
                    head_ = info['plan_raw'][:, :, d_:d_ + m_, :3].float().sum(2)                 # (k, N, 3)
                    tail_ = COH['prev'][:, R_ + d_:R_ + d_ + m_, :3].float().sum(1)                # (N, 3)
                    info['coh'] = (head_ - tail_[None]).norm(dim=-1).T                              # (N, k)
                    Qs = Q - COH['lam'] * info['coh']
                best = Qs.argmax(1)
                if EVID is not None and 'evid' in verifier_state:
                    sw_ = dist.T > EVID['theta']                                           # (N, k) switch candidates
                    sup_ch_ = torch.zeros_like(best, dtype=torch.bool)
                    if EVID.get('suppress', False):
                        perm_ = (verifier_state['evid']['c'] >= EVID.get('perm_k', EVID['kstar'])) | (verifier_state['evid']['cs'] >= EVID['fallback'])   # belief_parts_20261116: --evid-perm-k
                        info['evid_perm'] = perm_ & sw_.any(1)   # stallr_20261117: suppression lifted while a switch candidate exists
                        if EVID.get('soft', 0.0) > 0:   # soft_belief_20261113: distance penalty weighted by the missing evidence instead of a hard mask
                            w_ = (1.0 - verifier_state['evid']['c'].float() / max(float(EVID['kstar']), 1.0)).clamp(0, 1) * (~perm_).float()
                            b2_ = (Qs - EVID['soft'] * w_[:, None] * dist.T).argmax(1); sup_ch_ = b2_ != best; best = b2_
                        else:
                            sup_ = ~perm_ & (~sw_).any(1)
                            b2_ = torch.where(sup_, Qs.masked_fill(sw_, float('-inf')).argmax(1), best); sup_ch_ = b2_ != best; best = b2_
                    info['evid_sup'] = sup_ch_; info['evid_cs'] = verifier_state['evid']['cs']
                    if os.environ.get('RDP_QCONF') == '1':   # qconf_20261117: unanimous-ensemble condition on Q-chosen switches
                        pen_ = (Qs - Q)                                                     # continuity term (0 if off), (N, k)
                        Am_ = A_all + pen_[None]                                            # (M, N, k)
                        stay_ = ~sw_
                        bs_ = Qs.masked_fill(sw_, float('-inf')).argmax(1)                  # best stay candidate by the selection score
                        ar_ = torch.arange(best.shape[0], device=best.device)
                        unan_ = (Am_[:, ar_, best] > Am_[:, ar_, bs_]).all(0)              # every member prefers the chosen switch
                        drop_ = sw_[ar_, best] & stay_.any(1) & ~unan_
                        best = torch.where(drop_, bs_, best); info['qconf_drop'] = drop_
                    force_ = ((verifier_state['evid']['c'] >= EVID['kstar']) | (verifier_state['evid']['cs'] >= EVID.get('force_cs', 10 ** 9))) & sw_.any(1) & ~EVID_STATE['done']
                    best = torch.where(force_, Q.masked_fill(~sw_, float('-inf')).argmax(1), best)
                    EVID_STATE['done'] = EVID_STATE['done'] | sw_[torch.arange(sw_.shape[0], device=sw_.device), best]
                    info['evid_force'] = force_; info['evid_c'] = verifier_state['evid']['c']; info['evid_px'] = verifier_state['evid']['px']
                    info['evid_pr'] = verifier_state['evid']['pr']; info['evid_done'] = EVID_STATE['done'].clone()
            info['verifier_V'] = torch.sigmoid(v)
            info['verifier_P_used'] = P_in
            info['verifier_Q'] = Q
            if rule == 'learned_q':
                idx = best
            elif rule == 'explore_q':
                if 'cand_offset_mm' not in info or explore is None:
                    raise RuntimeError('explore_q needs --xgrid-mode with --xgrid-all-draws and --explore-eps')
                idx, info['explored'], info['explore_offset_mm'] = explore_offset_choice(
                    info['cand_offset_mm'], Q, float(explore['eps']), explore['rng'])
            else:
                idx = torch.where(info['verifier_V'] < verifier['tau'], best, idx)
    elif rule == 'dwell_rule':
        if 'plan_raw' not in info:
            raise RuntimeError('dwell_rule needs decoded plans (--record-plans)')
        if qt_extra is not None:                                                # step 0 has no history -> candidate 0
            qe = torch.as_tensor(np.asarray(qt_extra, np.float32), device=cands.device)
            con = qe[:, 22] > 0.5                                               # VMS H column 1 = in contact now
            dwell, zprog = qe[:, 35] * 100.0, qe[:, 36] * 10.0                  # in-contact steps at region, z descent mm
            mag = info['plan_raw'][..., :2].float().sum(2).mul(10.0).norm(dim=-1)   # (k, N) nominal plan xy displacement, mm
            idx = torch.where(con & (dwell < DWELL_STAY), mag.argmin(0), idx)
            idx = torch.where(con & (dwell >= DWELL_LEAVE) & (zprog < DWELL_ZPROG_MM), mag.argmax(0), idx)
    elif rule == 'term_guide':
        if 'plan_raw' not in info:
            raise RuntimeError('term_guide needs decoded plans (--record-plans)')
        tau = torch.zeros(n, device=cands.device) if switch_tau is None else torch.as_tensor(np.asarray(switch_tau, np.float32), device=cands.device)
        if GUIDE['force']:
            tau = torch.full((n,), GUIDE['tau_ref'], device=cands.device)
        fe = torch.nonzero(tau > 0).flatten()
        kg = k // 2
        gmask = torch.zeros(k, n, dtype=torch.bool, device=cands.device)
        t_g0 = time.time()
        if len(fe) and kg > 0:
            F = len(fe)
            Pv = info['plan_full'].reshape(k, n, -1).float()                       # normalized plans (k, N, H*A)
            med = Pv.median(0).values; sig = (Pv - med[None]).norm(dim=-1).median(0).values.clamp_min(1e-3)
            z0 = cands[kg:, fe].reshape(-1, cands.shape[-1]).float()                # (kg*F, D)
            ext_sub = {kk: vv[fe].repeat(kg, *([1] * (vv.dim() - 1))) for kk, vv in extended_obs_dict.items()}
            r = (GUIDE['rmax'] * torch.clamp(tau[fe] / GUIDE['tau_ref'], 0, 1)).repeat(kg)
            zg = _guide_escape(policy, z0, ext_sub, med[fe].repeat(kg, 1), sig[fe].repeat(kg), r, horizon, ratio, GUIDE['steps'], GUIDE['cap'])
            with torch.no_grad():
                key = next(iter(ext_sub)); hist = ext_sub[key]; acts = []
                for _ in range(horizon):
                    res = policy.predict_from_latent_action(zg.to(cands.dtype), {key: hist}, hist.shape[-2], ratio)
                    acts.append(res['action'][:, -1]); hist = torch.cat([hist, hist[:, -1:]], dim=1)
                rawg = torch.stack(acts, 1)                                          # (kg*F, H, A)
            cands = cands.clone(); cands[kg:, fe] = zg.view(kg, F, -1).to(cands.dtype)
            pr = info['plan_raw'].clone(); pr[kg:, fe] = rawg.view(kg, F, horizon, -1).to(pr.dtype); info['plan_raw'] = pr
            gmask[kg:, fe] = True
        info['guided_mask'] = gmask; info['guide_tau'] = tau; info['guide_seconds'] = time.time() - t_g0
        sw = info['plan_raw'][..., :2].float().sum(2).mul(10.0).norm(dim=-1) >= TERM_SW_MM       # (k, N)
        any_sw = sw.any(0); first_sw = torch.where(any_sw, sw.float().argmax(0), idx)
        if not GUIDE['measure_only']:
            go = (tau > 0) & any_sw
            if GUIDE['verifier'] is not None and term_x is not None and step_now is not None and step_now >= VER_S_MIN:
                # within the critic's mode, the verifier's argmax: SWITCH candidates where go, else STAY candidates (all if none)
                pr_ = info['plan_raw'].float(); cf = ver_cand_features(pr_)                         # (k, N, 11)
                xt = torch.as_tensor(np.asarray(term_x, np.float32), device=cands.device)            # (N, 62)
                Xv = torch.cat([xt[None].expand(k, -1, -1), cf], -1).reshape(k * n, -1).cpu().numpy().astype(np.float64)
                sc = torch.as_tensor(GUIDE['verifier'].predict(Xv).reshape(k, n), device=cands.device, dtype=torch.float32)
                stay_ok = ~sw; stay_ok = torch.where(stay_ok.any(0, keepdim=True), stay_ok, torch.ones_like(stay_ok))
                allowed = torch.where(go[None], sw, stay_ok)
                idx = sc.masked_fill(~allowed, -1e9).argmax(0)
                info['ver_score'] = sc
            else:
                idx = torch.where(go, first_sw, idx)
    elif rule == 'mode_first':
        if 'plan_raw' not in info or not MODE_SPACE:
            raise RuntimeError('mode_first needs decoded plans (--record-plans) and --mode-space')
        ms_ = MODE_SPACE; pr_ = info['plan_raw'].float().cpu().numpy().astype(np.float64)          # (k, N, H, A)
        f_ = (pr_ / ms_['sd']).reshape(k * n, -1)
        zz_ = (f_ - ms_['pca_mean']) @ ms_['pca_components'].T
        lab = ((zz_[:, None] - ms_['centers'][None]) ** 2).sum(-1).argmin(1).reshape(k, n)            # (k, N)
        km_ = ms_['centers'].shape[0]
        info['mode_lab'] = lab
        if mode_mu is not None and mode_gate is not None:
            M = np.asarray(mode_mu, np.float64).transpose(1, 0, 2)                                     # (NB, N, k)
            ok = ~np.isnan(M).any((0, 1)); M = np.where(np.isnan(M), 0.0, M)
            cnt = np.stack([(lab == c).sum(0) for c in range(km_)], 1)                                 # (N, k)
            oth = cnt - np.eye(km_)[lab[0]]
            P = (oth + 0.5) / (k - 1 + 0.5 * km_)
            w = P * ok[None]
            tau = M - (w[None] * M).sum(2, keepdims=True) / np.maximum(w.sum(1), 1e-9)[None, :, None]
            tp = np.where(ok[None] & (cnt > 0), tau.mean(0) - tau.std(0), -np.inf)                     # (N, k)
            cstar = tp.argmax(1); dev = (np.asarray(mode_gate) > 0) & (tp[np.arange(n), cstar] > 0)
            first_c = np.array([int(np.nonzero(lab[:, i] == cstar[i])[0][0]) for i in range(n)])
            idx = torch.as_tensor(np.where(dev, first_c, idx.cpu().numpy()), device=cands.device, dtype=idx.dtype)
            info['mode_tau'] = tp.astype(np.float32); info['mode_dev'] = dev
    elif rule in ('soft_tilt', 'soft_rimorc', 'soft_explore'):
        if 'plan_raw' not in info or 'eps0' not in MODE_SPACE:
            raise RuntimeError('soft_tilt needs decoded plans (--record-plans) and a --mode-space with share / sigma2 / eps0 / lmax')
        ms_ = MODE_SPACE; pr_ = info['plan_raw'].float().cpu().numpy().astype(np.float64)          # (k, N, H, A)
        zz_ = ((pr_ / ms_['sd']).reshape(k * n, -1) - ms_['pca_mean']) @ ms_['pca_components'].T
        d2_ = ((zz_[:, None] - ms_['centers'][None]) ** 2).sum(-1)                                     # (k*N, km)
        km_ = ms_['centers'].shape[0]; lab = d2_.argmin(1).reshape(k, n); info['mode_lab'] = lab
        lw_ = np.log(ms_['share'])[None] - d2_ / (2 * float(ms_['sigma2'])); lw_ -= lw_.max(1, keepdims=True)
        q_ = np.exp(lw_); q_ = (q_ / q_.sum(1, keepdims=True)).reshape(k, n, km_)
        if mode_mu is not None and mode_s is not None:
            M = np.asarray(mode_mu, np.float64).transpose(1, 0, 2)                                     # (NB, N, k)
            ok = ~np.isnan(M).any((0, 1)); M = np.where(np.isnan(M), 0.0, M)
            cnt = np.stack([(lab == c).sum(0) for c in range(km_)], 1)
            P = (cnt - np.eye(km_)[lab[0]] + 0.5) / (k - 1 + 0.5 * km_); w = P * ok[None]
            tau = M - (w[None] * M).sum(2, keepdims=True) / np.maximum(w.sum(1), 1e-9)[None, :, None]
            tp = np.where(ok[None], tau.mean(0) - tau.std(0), 0.0)                                      # (N, k)
            Ah = (q_ * np.maximum(tp, 0)[None]).sum(-1)                                                 # (k, N)
            sv = np.asarray(mode_s, np.float64); eps0, lmax = float(ms_['eps0']), float(ms_['lmax'])
            idx_np = idx.cpu().numpy().copy(); moved = np.zeros(n, bool); kls = np.zeros(n, np.float32); pis = np.full((n, k), 1.0 / k, np.float32)
            for i in range(n):
                a_ = Ah[:, i]; eps = eps0 * sv[i]
                if eps <= 0 or np.ptp(a_) < 1e-12:
                    continue
                def pk(l):
                    p = np.exp(l * (a_ - a_.max())); p /= p.sum(); return p, float((p * np.log(p * k + 1e-300)).sum())
                p, kl = pk(lmax)
                if kl >= eps:
                    lo, hi = 0.0, lmax
                    for _ in range(60):
                        mid = 0.5 * (lo + hi); _p, _kl = pk(mid)
                        lo, hi = (mid, hi) if _kl < eps else (lo, mid)
                    p, kl = pk(hi)
                kls[i] = kl; pis[i] = p
                rng_ = np.random.default_rng([int(soft_seed if soft_seed is not None else 0), int(req_seed), i])
                if rng_.random() >= min(1.0, k * p[0]):
                    res_ = np.maximum(p - 1.0 / k, 0.0); res_ /= res_.sum()
                    idx_np[i] = int(rng_.choice(k, p=res_)); moved[i] = idx_np[i] != 0
            idx = torch.as_tensor(idx_np, device=cands.device, dtype=idx.dtype)
            info['mode_tau'] = tp.astype(np.float32); info['mode_dev'] = moved; info['soft_kl'] = kls; info['soft_pi'] = pis
        if rule == 'soft_rimorc' and rim_orc is not None:
            ro = np.asarray(rim_orc, np.float64); fl = ro[:, 0] > 0
            la = np.linalg.norm(ro[None, :, 1:3] + 10.0 * pr_[..., :2].sum(2), axis=-1)                  # (k, N) mm
            best = la.argmin(0); idx_np = idx.cpu().numpy().copy(); ovr = fl & (best != idx_np)
            idx_np[fl] = best[fl]; idx = torch.as_tensor(idx_np, device=cands.device, dtype=idx.dtype)
            info['rim_flag'] = fl; info['rim_override'] = ovr
        if rule == 'soft_explore' and explore_k is not None:
            ek = np.asarray(explore_k, np.int64); idx_np = idx.cpu().numpy().copy(); info['explore_soft_idx'] = idx_np.copy()
            idx_np[ek >= 0] = ek[ek >= 0]; idx = torch.as_tensor(idx_np, device=cands.device, dtype=idx.dtype); info['explore_k'] = ek
    elif rule == 'dec_soft':
        if 'plan_raw' not in info or not DEC or 'eps0' not in MODE_SPACE or dec_feats is None or dec_x is None:
            raise RuntimeError('dec_soft needs --record-plans, --record-path (obs features), --dec-dir, --dec-curve, --mode-space and the bridge --send_dec')
        ms_ = MODE_SPACE; pr_ = info['plan_raw'].float().cpu().numpy().astype(np.float64)                  # (k, N, H, A)
        # 1) soft_tilt distribution exactly as SOFT16 (uniform where SOFT16 does not tilt)
        zz_ = ((pr_ / ms_['sd']).reshape(k * n, -1) - ms_['pca_mean']) @ ms_['pca_components'].T
        d2_ = ((zz_[:, None] - ms_['centers'][None]) ** 2).sum(-1)
        km_ = ms_['centers'].shape[0]; lab = d2_.argmin(1).reshape(k, n); info['mode_lab'] = lab
        lw_ = np.log(ms_['share'])[None] - d2_ / (2 * float(ms_['sigma2'])); lw_ -= lw_.max(1, keepdims=True)
        q_ = np.exp(lw_); q_ = (q_ / q_.sum(1, keepdims=True)).reshape(k, n, km_)
        psoft = np.full((n, k), 1.0 / k)
        if mode_mu is not None and mode_s is not None:
            M = np.asarray(mode_mu, np.float64).transpose(1, 0, 2)
            ok = ~np.isnan(M).any((0, 1)); M = np.where(np.isnan(M), 0.0, M)
            cnt = np.stack([(lab == c).sum(0) for c in range(km_)], 1)
            P = (cnt - np.eye(km_)[lab[0]] + 0.5) / (k - 1 + 0.5 * km_); w = P * ok[None]
            tau = M - (w[None] * M).sum(2, keepdims=True) / np.maximum(w.sum(1), 1e-9)[None, :, None]
            tp = np.where(ok[None], tau.mean(0) - tau.std(0), 0.0)
            Am = (q_ * np.maximum(tp, 0)[None]).sum(-1)
            sv = np.asarray(mode_s, np.float64); eps0, lmax = float(ms_['eps0']), float(ms_['lmax'])
            for i in range(n):
                a_ = Am[:, i]; eps = eps0 * sv[i]
                if eps <= 0 or np.ptp(a_) < 1e-12:
                    continue
                def pk(l):
                    p = np.exp(l * (a_ - a_.max())); p /= p.sum(); return p, float((p * np.log(p * k + 1e-300)).sum())
                p, kl = pk(lmax)
                if kl >= eps:
                    lo, hi = 0.0, lmax
                    for _ in range(60):
                        mid = 0.5 * (lo + hi); _p, _kl = pk(mid)
                        lo, hi = (mid, hi) if _kl < eps else (lo, mid)
                    p, kl = pk(hi)
                psoft[i] = p
            info['mode_tau'] = tp.astype(np.float32)
        # 2) decomposed verifier score
        dx_ = np.asarray(dec_x, np.float64); c_ = dx_[:, :3]; dg_ = dx_[:, 3].astype(np.int64); con_ = dx_[:, 4] > 0
        Ahat = dec_score(dec_feats, c_, dg_, pr_)                                                          # (k, N)
        pk_ = 0.01 * pr_[..., :3].sum(2)                                                                   # (k, N, 3)
        ret_ = (pk_ * c_[None]).sum(-1) < 0
        idx_np = np.zeros(n, np.int64); hz_ = np.zeros(n, bool); kld = np.zeros(n, np.float32); mv_ = np.zeros(n, bool); pfin = np.zeros((n, k), np.float32); pbas = np.zeros((n, k), np.float32)
        for i in range(n):
            p = psoft[i].copy(); R_ = ret_[:, i]
            if dg_[i] > 0 and R_.any() and not R_.all():
                ps_ = 1.0 / (1.0 + np.exp(-(DEC['a'] + DEC['b'] * min(int(dg_[i]), DEC['d_cap']))))
                p[R_] *= ps_ / p[R_].sum(); p[~R_] *= (1.0 - ps_) / p[~R_].sum(); hz_[i] = True
            base_ = p.copy(); pbas[i] = base_
            if con_[i] and step_now is not None and step_now >= 12:
                a_ = Ahat[:, i]
                def tilt(l):
                    q = np.zeros(k)
                    for cl_ in ((R_, ~R_) if hz_[i] else (np.ones(k, bool),)):
                        qc = base_[cl_] * np.exp(l * (a_[cl_] - a_[cl_].max())); q[cl_] = base_[cl_].sum() * qc / qc.sum()
                    return q, float((q * np.log(np.maximum(q, 1e-300) / np.maximum(base_, 1e-300))).sum())
                q, kl = tilt(DEC['lmax'])
                if kl > DEC['eps']:
                    lo, hi = 0.0, DEC['lmax']
                    for _ in range(60):
                        mid = 0.5 * (lo + hi); _q, _kl = tilt(mid)
                        lo, hi = (mid, hi) if _kl < DEC['eps'] else (lo, mid)
                    q, kl = tilt(lo)
                p = q; kld[i] = kl
            pfin[i] = p
            rng_ = np.random.default_rng([int(soft_seed if soft_seed is not None else 0), int(req_seed), i])
            if rng_.random() >= min(1.0, k * p[0]):
                res_ = np.maximum(p - 1.0 / k, 0.0)
                if res_.sum() > 0:
                    res_ /= res_.sum(); idx_np[i] = int(rng_.choice(k, p=res_)); mv_[i] = idx_np[i] != 0
        info['explore_base_idx'] = idx_np.copy()
        if explore_k is not None:
            ek = np.asarray(explore_k, np.int64); idx_np[ek >= 0] = ek[ek >= 0]; info['explore_k'] = ek
        else:
            info['explore_k'] = np.full(n, -1, np.int64)
        idx = torch.as_tensor(idx_np, device=cands.device, dtype=idx.dtype)
        info['dec_ahat'] = Ahat.astype(np.float32); info['dec_haz'] = hz_; info['dec_contact'] = con_; info['dec_kl'] = kld; info['dec_moved'] = mv_; info['dec_p'] = pfin; info['dec_base'] = pbas; info['dec_psoft'] = psoft.astype(np.float32)
    elif rule in ('haz', 'haz_soft'):
        if 'plan_raw' not in info or not HAZ:
            raise RuntimeError('haz needs decoded plans (--record-plans) and --haz-curve')
        lift_ = info['plan_raw'][..., 2].float().sum(2).cpu().numpy() > 0                          # (k, N)
        Ah = np.zeros((k, n))
        if rule == 'haz_soft':
            if 'eps0' not in MODE_SPACE:
                raise RuntimeError('haz_soft needs --mode-space (soft)')       # no mode_mu yet (first refresh): A_hat = 0, uniform within class
            ms_ = MODE_SPACE; pr_ = info['plan_raw'].float().cpu().numpy().astype(np.float64)          # (k, N, H, A)
            zz_ = ((pr_ / ms_['sd']).reshape(k * n, -1) - ms_['pca_mean']) @ ms_['pca_components'].T
            d2_ = ((zz_[:, None] - ms_['centers'][None]) ** 2).sum(-1)                                     # (k*N, km)
            km_ = ms_['centers'].shape[0]; lab = d2_.argmin(1).reshape(k, n); info['mode_lab'] = lab
            lw_ = np.log(ms_['share'])[None] - d2_ / (2 * float(ms_['sigma2'])); lw_ -= lw_.max(1, keepdims=True)
            q_ = np.exp(lw_); q_ = (q_ / q_.sum(1, keepdims=True)).reshape(k, n, km_)
            if mode_mu is not None:
                M = np.asarray(mode_mu, np.float64).transpose(1, 0, 2)                                     # (NB, N, k)
                ok = ~np.isnan(M).any((0, 1)); M = np.where(np.isnan(M), 0.0, M)
                cnt = np.stack([(lab == c).sum(0) for c in range(km_)], 1)
                P = (cnt - np.eye(km_)[lab[0]] + 0.5) / (k - 1 + 0.5 * km_); w = P * ok[None]
                tau = M - (w[None] * M).sum(2, keepdims=True) / np.maximum(w.sum(1), 1e-9)[None, :, None]
                tp = np.where(ok[None], tau.mean(0) - tau.std(0), 0.0)                                      # (N, k)
                Ah = (q_ * np.maximum(tp, 0)[None]).sum(-1)                                                 # (k, N)
                info['mode_tau'] = tp.astype(np.float32)
            info['mode_lab'] = lab
        dd_ = np.zeros(n, np.int64) if stall_d is None else np.asarray(stall_d, np.int64)
        lam_ = float(MODE_SPACE['lmax']) if rule == 'haz_soft' else 0.0
        idx_np = idx.cpu().numpy().copy(); tgt_ = np.zeros(n, np.float32); hmov_ = np.zeros(n, bool); smov_ = np.zeros(n, bool)
        sv_ = None if (rule != 'haz_soft' or mode_mu is None or mode_s is None) else np.asarray(mode_s, np.float64)
        for i in range(n):
            L_ = lift_[:, i]
            if dd_[i] <= 0 or L_.all() or not L_.any():
                # hazard not applicable: haz -> B16 (candidate 0); haz_soft -> soft_tilt exactly as SOFT16 (KL budget eps0 * s, same rng)
                if sv_ is not None:
                    a_ = Ah[:, i]; eps = float(MODE_SPACE['eps0']) * sv_[i]; lmax = float(MODE_SPACE['lmax'])
                    if eps <= 0 or np.ptp(a_) < 1e-12:
                        continue
                    def pk(l):
                        p = np.exp(l * (a_ - a_.max())); p /= p.sum(); return p, float((p * np.log(p * k + 1e-300)).sum())
                    p, kl = pk(lmax)
                    if kl >= eps:
                        lo, hi = 0.0, lmax
                        for _ in range(60):
                            mid = 0.5 * (lo + hi); _p, _kl = pk(mid)
                            lo, hi = (mid, hi) if _kl < eps else (lo, mid)
                        p, kl = pk(hi)
                    rng_ = np.random.default_rng([int(soft_seed if soft_seed is not None else 0), int(req_seed), i])
                    if rng_.random() >= min(1.0, k * p[0]):
                        res_ = np.maximum(p - 1.0 / k, 0.0); res_ /= res_.sum()
                        idx_np[i] = int(rng_.choice(k, p=res_)); smov_[i] = idx_np[i] != 0
                continue
            ps_ = 1.0 / (1.0 + np.exp(-(HAZ['a'] + HAZ['b'] * min(int(dd_[i]), HAZ['d_cap'])))); tgt_[i] = ps_
            p_ = np.zeros(k)
            for cls_, mass_ in ((L_, ps_), (~L_, 1.0 - ps_)):
                a_ = Ah[cls_, i]; e_ = np.exp(lam_ * (a_ - a_.max())); p_[cls_] = mass_ * e_ / e_.sum()
            rng_ = np.random.default_rng([int(soft_seed if soft_seed is not None else 0), int(req_seed), i, 7])
            b0_ = int(idx_np[i])
            if rng_.random() >= min(1.0, k * p_[b0_]):
                res_ = np.maximum(p_ - 1.0 / k, 0.0); res_ /= res_.sum()
                idx_np[i] = int(rng_.choice(k, p=res_)); hmov_[i] = idx_np[i] != b0_
        idx = torch.as_tensor(idx_np, device=cands.device, dtype=idx.dtype)
        info['haz_d'] = dd_.astype(np.int32); info['haz_target'] = tgt_; info['haz_nlift'] = lift_.sum(0).astype(np.int8); info['haz_moved'] = hmov_; info['haz_soft_moved'] = smov_; info['haz_ahat'] = Ah.astype(np.float32)
    elif rule == 'stop_first':
        if 'plan_raw' not in info:
            raise RuntimeError('stop_first needs decoded plans (--record-plans)')
        stp = info['plan_raw'][..., 2].float().sum(2).mul(10.0) > STOP_MM                       # (k, N)
        first_stop = torch.where(stp.any(0), stp.float().argmax(0), idx)
        if stop_flag is not None:
            fl = torch.as_tensor(np.asarray(stop_flag) > 0, device=cands.device)
            idx = torch.where(fl & stp.any(0), first_stop, idx)
    elif rule in ('rand_switch', 'term_switch'):
        if 'plan_raw' not in info:
            raise RuntimeError(rule + ' needs decoded plans (--record-plans)')
        sw = info['plan_raw'][..., :2].float().sum(2).mul(10.0).norm(dim=-1) >= TERM_SW_MM       # (k, N)
        any_sw = sw.any(0); first_sw = torch.where(any_sw, sw.float().argmax(0), idx)
        if rule == 'term_switch':
            if switch_flag is not None:
                fl = torch.as_tensor(np.asarray(switch_flag) > 0, device=cands.device)
                idx = torch.where(fl & any_sw, first_sw, idx)
        elif qt_extra is not None:
            qe = torch.as_tensor(np.asarray(qt_extra, np.float32), device=cands.device)
            elig = (qe[:, 22] > 0.5) & (qe[:, 20] * 499.0 >= 24.0) & any_sw
            u = torch.as_tensor(rng.random(n), device=cands.device) < TERM_EPS
            pick = torch.as_tensor(rng.random((k, n)), device=cands.device).masked_fill(~sw, -1.0).argmax(0)
            idx = torch.where(elig & u, pick, idx)
    elif rule == 'oracle_endpoint':
        # PRIVILEGED selection ceiling (multisocket_orc_20261019, diagnostic only): see the module docstring.
        if privileged_offset is None or 'plan_raw' not in info:
            raise RuntimeError('oracle_endpoint requires decoded plans and privileged_offset (N, 3) from the bridge')
        po = torch.as_tensor(np.asarray(privileged_offset, np.float32), device=cands.device)
        if po.dim() != 2 or po.shape[1] != 3:
            raise RuntimeError('oracle_endpoint needs privileged_offset (N, 3) = [dx, dy, gate]')
        disp = info['plan_raw'][..., :2].float().sum(2) * 10.0                                  # (k, N, 2) mm
        err = (disp - po[None, :, :2] * 1000.0).norm(dim=-1)                                    # (k, N)
        info['oracle_err_mm'] = err
        idx = torch.where(po[:, 2] > 0.5, err.argmin(0), idx)
    elif rule == 'oracle_land':
        # PRIVILEGED upper bound (diagnostic only): candidate whose executed-window displacement best closes the
        # socket-plug offset, |offset - rho * disp_xy| -- the same pred_land_err_mm used in the selection records.
        if privileged_offset is None or 'cand_exec_disp_mm' not in info:
            raise RuntimeError('oracle_land requires --xgrid-mode and privileged_offset (--send_privileged_offset)')
        off = torch.as_tensor(privileged_offset, device=cands.device, dtype=torch.float32) * 1000.0     # (N, 2) mm
        err = (off[None] - xgrid['rho'] * info['cand_exec_disp_mm'].float()[..., :2]).norm(dim=-1)   # (k, N)
        idx = err.argmin(0)
    elif rule == 'oracle_phase':
        # PRIVILEGED phase-aware upper bound (diagnostic only, obs_recovery 2026-10-11): where the plug is tilted & in contact,
        # the candidate maximizing z-scored (8-step lift) + z-scored (rotation along the upright axis); elsewhere oracle_direction.
        if 'align' not in info or 'tilt_axis' not in info:
            raise RuntimeError('oracle_phase requires --send_privileged_offset --send_privileged_tilt on the client')
        pr = info['plan_raw'].float()                                        # (k, N, H, A) unnormalized
        ax = info['tilt_axis'].float()                                       # (N, 3)
        zs = lambda x: (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
        rec = (zs(pr[..., 2].sum(-1)) + zs((pr[..., 3:6].sum(2) * ax[None]).sum(-1))).argmax(0)
        tilted = ax.norm(dim=-1) > 0.5
        idx = torch.where(tilted, rec, info['align'].argmax(0))
        info['phase_tilted'] = tilted
    elif rule in ('oracle_direction_when_stuck', 'anti_direction_when_stuck', 'oracle_direction'):
        if 'align' not in info:
            raise RuntimeError(rule + ' requires privileged_offset from the client (--send_privileged_offset)')
        best = info['align'].argmax(0) if rule.startswith('oracle') else info['align'].argmin(0)
        idx = best if rule == 'oracle_direction' else torch.where(stuck, best, idx)   # oracle_direction: every refresh (candidate-set upper bound)
    elif rule in ('consistent', 'inconsistent', 'inconsistent_when_stuck') and have_hist:
        d = info['dist']
        if rule == 'consistent':
            idx = d.argmin(0)
        else:
            idx = d.argmax(0)
            if rule == 'inconsistent_when_stuck':
                idx = torch.where(stuck, idx, torch.zeros_like(idx))
    if record_vr and raw_xy is not None and 'dist' in info and 'cand_offset_mm' in info:
        info['vr_cand'] = vr_cand_features(info['plan_mean'], info['lateral'], info['dist'], raw_xy,
                                           info['cand_offset_mm'], info['cand_exec_disp_mm'])        # (k, N, 19)
    info['cands'] = cands
    if 'act_offset' in info:
        info['chosen_act_offset'] = info['act_offset'][idx, ar]
    if 'plan_raw' in info: COH['prev'] = info['plan_raw'][idx, ar].detach()
    return cands[idx, ar], idx, info


RECORD_PLANS = False
from latent_steer import steer_coeffs, apply_steer   # noqa: E402 (opt-in steering helpers)


def serve(policy, constants, port, device, latent_mode='normal', select_rule='none', select_k=8, select_horizon=8,
          record_path=None, verifier=None, residual=None, refine_steps=5, refine_radius=0.25,
          freeze_tactile=False, select_every=1, select_pool=0, head=None, local_radius=0.0,
          draw_chunk=4, codebook=None, local_keep=0, local_cov=None, two_stage_keep=0, two_stage_radius=0.1,
          xgrid=None, feat_encoder=None, explore=None, record_vr=False, need_tactile=False,
          draw_noise_mode='global', draw_noise_base=8, record_timing=False, early_records=4, cache_cond=False, steer=None):
    device = torch.device(device)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', port))
    server.listen(1)
    print(f"[policy_server] listening on 127.0.0.1:{port}", file=sys.stderr)

    while True:
        conn, addr = server.accept()
        print(f"[policy_server] client connected: {addr}", file=sys.stderr)
        cached_latent = None
        dly_old = None; dly_new = None; dly_ext = None
        cached_act_offset = None      # xgrid 'action' mode: per-env x offset added to every executed action
        refresh_idx = -1
        frozen_ext = None
        frozen_age = 0
        recent_actions = collections.deque(maxlen=select_horizon)
        z_history = collections.deque(maxlen=64)
        pose_history = collections.deque(maxlen=64)
        recent12 = collections.deque(maxlen=12)
        step_counter = 0
        EVID_STATE.clear()
        gate_on = xgrid is not None and xgrid.get('gate', 'none') != 'none'
        gate_first = []           # tactile_stall gate: c8d[:3] of the first 5 requests of the episode (baseline)
        tac_first = []            # vretrain_v1 (opt-in): full c8d of the first 5 requests (baseline)
        tac_hist = collections.deque(maxlen=12)   # vretrain_v1: c8d of the last 12 requests
        if explore is not None:
            explore = dict(explore)
        records = collections.defaultdict(list)
        last_seed = 0
        select_stats = {'refreshes': 0, 'switched_from_first': 0, 'chosen_dist': 0.0, 'first_dist': 0.0, 'mean_dist': 0.0, 'scored': 0}
        try:
            send_msg(conn, constants)
            while True:
                req = recv_msg(conn)
                if req.get('cmd') == 'close':
                    break
                if 'policy_seed' in req:
                    torch.manual_seed(int(req['policy_seed']))
                    last_seed = int(req['policy_seed'])
                obs_dict = to_torch(req['obs'], device)
                extended_obs_dict = to_torch(req['extended_obs'], device)
                step_feats = None
                if req.get('want_feats'):
                    with torch.no_grad():
                        step_feats = obs_features(policy, obs_dict, feat_encoder).half().cpu().numpy()
                if 'history_replay' in req:
                    if step_counter != 0:
                        raise ValueError('history replay only allowed before first prediction')
                    history = req['history_replay']
                    ha = torch.as_tensor(history['actions'], device=device, dtype=torch.float32)
                    hp = torch.as_tensor(history['positions'], device=device, dtype=torch.float32)
                    t = int(history['step'])
                    n = obs_dict['state'].shape[0]
                    if ha.shape != (min(t, max(12, select_horizon)), n, policy.normalizer['action'].get_input_stats()['mean'].numel()) or hp.shape != (min(t, 64), n, 3):
                        raise ValueError('replay history dimensions do not match step/batch')
                    if not torch.isfinite(ha).all() or not torch.isfinite(hp).all():
                        raise ValueError('nonfinite replay history')
                    recent_actions.extend(ha[-select_horizon:].unbind(0))
                    recent12.extend(ha[-12:].unbind(0))
                    pose_history.extend(hp.unbind(0))
                    z_history.extend(hp[..., 2].unbind(0))
                    step_counter = t
                    refresh_idx = int(history['refresh_index']) - 1
                if gate_on and step_counter < 5 and len(gate_first) == step_counter and 'contact_8d_right' in extended_obs_dict:
                    gate_first.append(extended_obs_dict['contact_8d_right'][:, -1, :3].float().clone())
                if need_tactile and 'contact_8d_right' in extended_obs_dict:
                    c8_now = extended_obs_dict['contact_8d_right'][:, -1].float().clone()
                    if step_counter < 5 and len(tac_first) == step_counter:
                        tac_first.append(c8_now)
                    tac_hist.append(c8_now)
                with torch.no_grad():
                    if req.get('cmd') == 'sample_pool':
                        count = int(req['n_candidates'])
                        if not 1 <= count <= 16:
                            raise ValueError('n_candidates must be between 1 and 16')
                        candidates = []
                        for _ in range(count):
                            sample = policy.predict_action(
                                obs_dict,
                                dataset_obs_temporal_downsample_ratio=constants['obs_temporal_downsample_ratio'],
                                extended_obs_dict=extended_obs_dict,
                                return_latent_action=True)
                            candidates.append(sample['action'][:, 0, :].detach().cpu())
                        send_msg(conn, {'latents': torch.stack(candidates).numpy()})
                        continue
                    # 'real' has to go through the two-stage API: the single-shot
                    # path calls get_temporal_cond with no length, which only works
                    # when the client padded to original_horizon.
                    mode = constants.get('extended_obs_mode')
                    if mode == 'growing':
                        n_ext = next(iter(extended_obs_dict.values())).shape[-2]
                        if 'delay_base_n' in req and int(req.get('chunk_step', 0)) == 0 and cached_latent is not None:
                            dly_old = (cached_latent.clone(), None if cached_act_offset is None else cached_act_offset.clone()); dly_ext = extended_obs_dict
                            extended_obs_dict = {kk: vv[:, -int(req['delay_base_n']):] for kk, vv in extended_obs_dict.items()}; n_ext = next(iter(extended_obs_dict.values())).shape[-2]
                        if 'latent_override' in req:
                            cached_latent = torch.as_tensor(req['latent_override'], device=device)
                            cached_act_offset = None
                        elif int(req.get('chunk_step', 0)) == 0 and select_rule != 'none':
                            t_feat0 = _tsync(record_timing, obs_dict) if record_timing else None
                            draw_noise = None
                            if draw_noise_mode == 'per_draw_seeded':
                                if 'policy_seed' not in req:
                                    raise ValueError('--draw-noise per_draw_seeded needs the bridge --policy_seed')
                                draw_noise = {'mode': draw_noise_mode, 'seed': last_seed, 'base': int(draw_noise_base)}
                            rng = np.random.default_rng(last_seed)
                            ratio_ = constants['obs_temporal_downsample_ratio']
                            feats = None
                            if record_path is not None or verifier is not None or residual is not None:
                                feats = obs_features(policy, obs_dict, feat_encoder)
                            verifier_state = None
                            if (verifier is not None or residual is not None) and len(recent12) == 12 and step_counter >= 36 and len(z_history) >= 24 and 'state' in obs_dict:
                                stt = obs_dict['state'][:, -1]                   # ee xyz + quat at this request (= after action t-1)
                                acts = torch.stack(list(recent12), 1)             # (N, 12, A) executed actions t-12..t-1
                                astd = (acts - acts.mean(1, keepdim=True)).pow(2).mean(1).sqrt()
                                tcol = torch.full((stt.shape[0], 1), step_counter / 499.0, device=stt.device)
                                P = torch.cat([stt[:, :7], (z_history[-24] - stt[:, 2])[:, None], acts.mean(1), astd, tcol], 1)
                                d24 = stt[:, :3] - pose_history[-24]
                                d12 = stt[:, :3] - pose_history[-12]
                                Pg = torch.cat([stt[:, :7], d24, d24.norm(dim=-1, keepdim=True), d12, d12.norm(dim=-1, keepdim=True),
                                                acts.mean(1), astd, tcol], 1)
                                verifier_state = {'obs_features': feats.float(), 'P': P.float(), 'P_generic': Pg.float()}
                                if EVID is not None:
                                    x_ = torch.cat([feats.half().float(), stt[:, :7].float()], 1)
                                    z_ = ((x_[:, EVID['k']] - EVID['mu']) / EVID['sd']).clamp(-10, 10)
                                    px_ = z_ @ EVID['wx'] + EVID['bx']; pr_ = z_ @ EVID['wr'] + EVID['br']
                                    if EVID.get('ext', False):   # link_cl: evidence from the residual's own convergence head (margins sent by the bridge)
                                        if 'evid_ext' in req:
                                            e_ = torch.as_tensor(np.asarray(req['evid_ext'], np.float32), device=px_.device); px_ = e_[:, 0] + 0.5; pr_ = e_[:, 1] + 2.9
                                        else:
                                            px_ = torch.full_like(px_, 9.0); pr_ = torch.full_like(pr_, 9.0)
                                    _sr_ = os.environ.get('RDP_STALL_RATIO')   # stallr_20261117
                                    if _sr_ and len(recent_actions) > 0:
                                        _cmd_ = torch.stack(list(recent_actions), 0)[:, :, :3].float().sum(0).norm(dim=-1) * 0.01
                                        _stall_ = (Pg[:, 10].float() / _cmd_.clamp_min(1e-6)) < float(_sr_)
                                    else:
                                        _stall_ = Pg[:, 10].float() < 0.003
                                    ok_ = _stall_ & (px_ < 0.5) & (pr_ < 2.9)
                                    if 'c' not in EVID_STATE:
                                        EVID_STATE['c'] = torch.zeros_like(px_, dtype=torch.long); EVID_STATE['done'] = torch.zeros_like(ok_)
                                    EVID_STATE['c'] = torch.where(ok_, EVID_STATE['c'] + 1, torch.zeros_like(EVID_STATE['c']))
                                    st_ = _stall_
                                    EVID_STATE['cs'] = torch.where(st_, EVID_STATE.get('cs', torch.zeros_like(EVID_STATE['c'])) + 1, torch.zeros_like(EVID_STATE['c']))
                                    verifier_state['evid'] = {'c': EVID_STATE['c'].clone(), 'cs': EVID_STATE['cs'].clone(), 'px': px_, 'pr': pr_}
                                if 'qt_extra' in req:
                                    verifier_state['QT'] = torch.as_tensor(np.asarray(req['qt_extra'], np.float32), device=stt.device)
                                if need_tactile and len(tac_first) == 5 and len(tac_hist) > 0:
                                    verifier_state['T'] = vr_tactile_features(tac_hist, torch.stack(tac_first).mean(0)).float()
                            if explore is not None:
                                # independent of torch RNG (diffusion draws unchanged); reproducible per (seed, step)
                                explore['rng'] = np.random.default_rng([int(explore['seed']), int(last_seed), int(step_counter)])
                            k_stage1 = two_stage_keep if two_stage_keep > 0 else select_k
                            cached_latent, idx, info = select_latent(policy, constants, obs_dict, extended_obs_dict,
                                                                     select_rule, k_stage1, select_horizon, recent_actions,
                                                                     z_history=z_history, rng=rng, record=record_path is not None,
                                                                     privileged_offset=req.get('privileged_offset'), qt_extra=req.get('qt_extra'), switch_flag=req.get('switch_flag'), switch_tau=req.get('switch_tau'), term_x=req.get('term_x'), step_now=step_counter, stop_flag=req.get('stop_flag'), mode_mu=req.get('mode_mu'), mode_gate=req.get('mode_gate'), mode_s=req.get('mode_s'), soft_seed=req.get('soft_seed'), req_seed=last_seed, rim_orc=req.get('rim_orc'), explore_k=req.get('explore_k'), stall_d=req.get('stall_d'), dec_feats=(None if feats is None else feats.half().float().cpu().numpy().astype(np.float64)), dec_x=req.get('dec_x'),
                                                                     verifier=verifier, verifier_state=verifier_state,
                                                                     residual=residual, refine_steps=refine_steps,
                                                                     refine_radius=refine_radius,
                                                                     select_every=select_every, refresh_idx=refresh_idx + 1,
                                                                     select_pool=select_pool, head=head, local_radius=local_radius,
                                                                     draw_chunk=draw_chunk, codebook=codebook, local_keep=local_keep, local_cov=local_cov,
                                                                     given_cands=(PREGEN['c'] if (req.get('use_pregen') and PREGEN['c'] is not None) else None),
                                                                     xgrid=xgrid, explore=explore, record_vr=record_vr,
                                                                     gate_base=torch.stack(gate_first).mean(0) if len(gate_first) == 5 else None,
                                                                     draw_noise=draw_noise, timing=record_timing,
                                                                     cache_cond=cache_cond, ext_sign=req.get('xgrid_sign'))
                            if record_timing:
                                t_end_ = _tsync(True, obs_dict)
                                info['timing'] = {'t_features_s': info['t_mark_start'] - t_feat0,
                                                  't_diffusion_s': info['t_mark_gen'] - info['t_mark_start'],
                                                  't_decode_offset_s': info['t_mark_dec'] - info['t_mark_gen'],
                                                  't_score_s': t_end_ - info['t_mark_dec'],
                                                  't_select_latent_s': t_end_ - info['t_mark_start'],
                                                  't_refresh_total_s': t_end_ - t_feat0}
                            cached_act_offset = info.get('chosen_act_offset')
                            if two_stage_keep > 0 and verifier_state is not None and select_rule == 'learned_q':
                                # Stage 2 -- derivative-free local search: Q is asked only to RANK. The stage-1
                                # winner stays in the set, so a bad perturbation costs nothing; the perturbations
                                # sit at the policy's own fine-variation scale (r = 0.1 ||z||), where random
                                # candidates matched fresh draws (73-76% vs 69%) while a gradient ascent on the
                                # same critic did not (exploitation).
                                z1 = cached_latent
                                d = torch.randn(select_k - 1, *z1.shape, device=z1.device)
                                d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                                pert = z1[None] + two_stage_radius * z1.norm(dim=-1, keepdim=True)[None] * d
                                cands2 = torch.cat([z1[None], pert], 0)
                                stage1_idx = idx
                                cached_latent, idx, info = select_latent(policy, constants, obs_dict, extended_obs_dict,
                                                                         select_rule, select_k, select_horizon, recent_actions,
                                                                         z_history=z_history, rng=rng, record=record_path is not None,
                                                                         privileged_offset=req.get('privileged_offset'), qt_extra=req.get('qt_extra'), switch_flag=req.get('switch_flag'),
                                                                         verifier=verifier, verifier_state=verifier_state,
                                                                         given_cands=cands2)
                                info['stage1_idx'] = stage1_idx
                            if steer is not None:
                                # latent_steer_20261012 (opt-in): bounded learned/explored edit of the selected latent
                                # (or a per-window action residual); only where learned_q itself is active
                                t_st0 = _tsync(record_timing, obs_dict) if record_timing else None
                                st_on = verifier_state is not None and 'T' in verifier_state
                                n_ = cached_latent.shape[0]
                                st_a = torch.full((n_, steer['m']), float('nan'), device=cached_latent.device)
                                z0_ = cached_latent
                                if st_on:
                                    st_a = steer_coeffs(steer, feats, verifier_state['P'], verifier_state['T'],
                                                        z0_, last_seed, step_counter)
                                    cached_latent, r_ = apply_steer(steer, st_a, z0_)
                                    if r_ is not None:
                                        cached_act_offset = r_ if cached_act_offset is None else cached_act_offset + r_
                                if record_timing:
                                    info.setdefault('timing', {})['t_steer_s'] = _tsync(True, obs_dict) - t_st0
                                if record_path is not None:
                                    records['steer_step'].append(step_counter)
                                    records['steer_on'].append(bool(st_on))
                                    records['steer_a'].append(st_a.float().cpu().numpy())
                                    records['steer_z0'].append(z0_.float().cpu().numpy())
                                    nanf_ = lambda d_: np.full((n_, d_), np.nan, np.float32)
                                    records['steer_P'].append(verifier_state['P'].float().cpu().numpy() if st_on else nanf_(21))
                                    records['steer_T'].append(verifier_state['T'].float().cpu().numpy() if st_on else nanf_(VR_TACTILE_DIM))
                            refresh_idx += 1
                            select_stats['refreshes'] += 1
                            select_stats['switched_from_first'] += int((idx != 0).sum())
                            select_stats['stuck_flags'] = select_stats.get('stuck_flags', 0) + int(info['stuck'].sum()); select_stats['z_hist_len'] = len(z_history)
                            if 'gate_trigger' in info:
                                select_stats['gate_trigger'] = select_stats.get('gate_trigger', 0) + int(info['gate_trigger'].sum())
                                select_stats['gate_contact'] = select_stats.get('gate_contact', 0) + int(info['gate_contact'].sum())
                            if 'dist' in info:
                                ar = torch.arange(idx.shape[0], device=idx.device)
                                dist = info['dist']
                                select_stats['scored'] += dist.shape[1]
                                select_stats['chosen_dist'] += float(dist[idx, ar].sum())
                                select_stats['first_dist'] += float(dist[0].sum())
                                select_stats['mean_dist'] += float(dist.mean(0).sum())
                            if record_path is not None:
                                if 'verifier_V' in info:
                                    records['verifier_P'].append(info['verifier_P_used'].float().cpu().numpy())
                                    records['verifier_step'].append(step_counter)
                                    records['verifier_V'].append(info['verifier_V'].float().cpu().numpy())
                                    records['verifier_Q'].append(info['verifier_Q'].float().cpu().numpy())
                                    if 'evid_c' in info:
                                        for kk_e in ('evid_c', 'evid_px', 'evid_pr', 'evid_force', 'evid_done', 'evid_cs', 'evid_sup'):
                                            records[kk_e].append(info[kk_e].float().cpu().numpy())
                                        if 'qconf_drop' in info:
                                            records['qconf_drop'].append(info['qconf_drop'].float().cpu().numpy())
                                        if 'evid_perm' in info:   # stallr_20261117
                                            records['evid_perm'].append(info['evid_perm'].float().cpu().numpy())
                                records['step'].append(step_counter)
                                if 'timing' in info:   # opt-in --record-timing (seconds, CUDA-synchronized)
                                    for kk_t, vv_t in info['timing'].items():
                                        records[kk_t].append(float(vv_t))
                                if draw_noise is not None:
                                    records['draw_noise_seed'].append(int(last_seed))
                                    records['draw_calls'].append(int(info.get('draw_calls', -1)))
                                    records['draw_oom'].append(int(info.get('draw_oom', 0)))
                                    if refresh_idx < early_records:   # full-precision shared originals for bitwise checks
                                        records['early_step'].append(step_counter)
                                        records['early_base_draws'].append(info['base_draws'].float().cpu().numpy())
                                        records['early_obs_features'].append(feats.float().cpu().numpy())
                                        records['early_chosen_latent'].append(cached_latent.float().cpu().numpy())
                                records['obs_features'].append(feats.half().cpu().numpy())
                                records['cand_latents'].append(info['cands'].half().cpu().numpy())
                                if RECORD_PLANS:   # opt-in (2026-10-10 recovery-candidate probe): raw decoded plans of all k candidates
                                    records['plan_raw'].append(info['plan_raw'].float().cpu().numpy() if 'plan_raw' in info
                                                               else np.full((info['cands'].shape[0], info['cands'].shape[1], select_horizon, 6), np.nan, np.float32))
                                records['chosen'].append(idx.cpu().numpy())
                                records['candidate_source'].append(info['candidate_source'].cpu().numpy())
                                records['stuck'].append(info['stuck'].cpu().numpy())
                                if 'mode_lab' in info:
                                    records['mode_lab'].append(np.asarray(info['mode_lab'], np.int8))
                                    if 'mode_dev' in info:
                                        records['mode_tau'].append(info['mode_tau']); records['mode_dev'].append(info['mode_dev'])
                                        if 'soft_kl' in info:
                                            records['soft_kl'].append(info['soft_kl']); records['soft_pi'].append(info['soft_pi'])
                                        if 'rim_flag' in info:
                                            records['rim_flag'].append(info['rim_flag']); records['rim_override'].append(info['rim_override'])
                                        if 'explore_k' in info:
                                            records['explore_k'].append(info['explore_k']); records['explore_soft_idx'].append(info['explore_soft_idx'])
                                if 'dec_ahat' in info:
                                    for kk_ in ('dec_ahat', 'dec_haz', 'dec_contact', 'dec_kl', 'dec_moved', 'dec_p', 'dec_base', 'dec_psoft'):
                                        records[kk_].append(info[kk_])
                                    if 'explore_base_idx' in info:
                                        records['explore_k'].append(info['explore_k']); records['explore_base_idx'].append(info['explore_base_idx'])
                                if 'haz_target' in info:
                                    for kk_ in ('haz_d', 'haz_target', 'haz_nlift', 'haz_moved', 'haz_soft_moved', 'haz_ahat'):
                                        records[kk_].append(info[kk_])
                                if 'guided_mask' in info:
                                    records['guided_mask'].append(info['guided_mask'].cpu().numpy())
                                    records['guide_tau'].append(info['guide_tau'].float().cpu().numpy())
                                    records['guide_seconds'].append(np.float32(info['guide_seconds']))
                                if GUIDE['verifier'] is not None and 'plan_raw' in info:
                                    records['ver_score'].append(info['ver_score'].cpu().numpy() if 'ver_score' in info
                                                                else np.full(info['plan_raw'].shape[:2], np.nan, np.float32))
                                if 'ext_sign' in info:
                                    records['xgrid_sign'].append(info['ext_sign'].cpu().numpy())
                                if 'gate_trigger' in info:
                                    records['gate_trigger'].append(info['gate_trigger'].cpu().numpy())
                                    records['gate_contact'].append(info['gate_contact'].cpu().numpy())
                                    records['gate_contact_ind'].append(info['gate_contact_ind'].float().cpu().numpy())
                                records['has_history'].append('dist' in info)
                                if record_vr:   # opt-in verifier-retrain inputs; NaN where not yet available
                                    kk_, nn_ = info['cands'].shape[:2]
                                    vs_ = verifier_state if verifier_state is not None else {}
                                    nanf = lambda *sh: np.full(sh, np.nan, np.float32)
                                    records['vr_valid'].append(verifier_state is not None and 'T' in vs_ and 'vr_cand' in info)
                                    records['vr_P'].append(vs_['P'].cpu().numpy() if 'P' in vs_ else nanf(nn_, 21))
                                    records['vr_T'].append(vs_['T'].cpu().numpy() if 'T' in vs_ else nanf(nn_, VR_TACTILE_DIM))
                                    records['vr_cand'].append(info['vr_cand'].float().cpu().numpy() if 'vr_cand' in info
                                                              else nanf(kk_, nn_, VR_CAND_DIM))
                                    records['vr_explored'].append(info['explored'].cpu().numpy() if 'explored' in info
                                                                  else np.zeros(nn_, bool))
                                    records['vr_explore_offset_mm'].append(info['explore_offset_mm'].cpu().numpy()
                                                                           if 'explore_offset_mm' in info else np.zeros(nn_, np.float32))
                                if 'residual_norm' in info:
                                    records['residual_norm'].append(info['residual_norm'].float().cpu().numpy())
                                if 'refine_move' in info:
                                    records['refine_move'].append(info['refine_move'].float().cpu().numpy())
                                if 'plan_mean' in info:   # the residual rule decodes no plans
                                    records['cand_plan_mean'].append(info['plan_mean'].float().cpu().numpy())
                                    records['cand_plan_full'].append(info['plan_full'].half().cpu().numpy())
                                    records['cand_lateral'].append(info['lateral'].float().cpu().numpy())
                                    records['dist'].append(info['dist'].float().cpu().numpy() if 'dist' in info
                                                           else np.full(info['lateral'].shape, np.nan, np.float32))
                                    if 'align' in info:
                                        records['align'].append(info['align'].float().cpu().numpy())
                                        records['offset'].append(np.asarray(req['privileged_offset'], np.float32)[:, :2])
                                    if 'phase_tilted' in info:
                                        records['phase_tilted'].append(info['phase_tilted'].cpu().numpy())
                                if 'cand_exec_disp_mm' in info:
                                    # per-candidate executed-window plan displacement (nominal mm), its grid offset,
                                    # and -- with the privileged socket-plug offset -- predicted landing error using
                                    # the preregistered realized/nominal ratio rho: |offset - rho * disp_xy| (mm).
                                    ed = info['cand_exec_disp_mm'].float()
                                    records['cand_exec_disp_mm'].append(ed.cpu().numpy())
                                    records['cand_offset_mm'].append(info['cand_offset_mm'].float().cpu().numpy())
                                    records['ee_state'].append(obs_dict['state'][:, -1].float().cpu().numpy())
                                    if 'clamp_target' in info:
                                        for kk in ('clamp_target', 'clamp_achieved', 'cand_offaxis', 'cand_dz_norm', 'cand_exec_sum_full'):
                                            records[kk].append(info[kk].float().cpu().numpy())
                                        records['diff_time_draws_s'].append(float(info['diff_time_draws_s']))
                                        records['diff_time_clamp_s'].append(float(info['diff_time_clamp_s']))
                                    if req.get('privileged_offset') is not None:
                                        off = torch.as_tensor(np.asarray(req['privileged_offset'], np.float32)[:, :2], device=ed.device) * 1000.0
                                        records['pred_land_err_mm'].append((off[None] - xgrid['rho'] * ed[..., :2]).norm(dim=-1).cpu().numpy())
                        elif int(req.get('chunk_step', 0)) == 0:
                            lat_res = policy.predict_action(
                                obs_dict,
                                dataset_obs_temporal_downsample_ratio=constants['obs_temporal_downsample_ratio'],
                                extended_obs_dict=extended_obs_dict,
                                return_latent_action=True)
                            cached_latent = lat_res['action'][:, 0, :]
                            cached_act_offset = None
                            if latent_mode == 'shuffle':
                                cached_latent = cached_latent[torch.randperm(
                                    cached_latent.shape[0], device=cached_latent.device)]
                            elif latent_mode == 'batchmean':
                                cached_latent = cached_latent.mean(
                                    dim=0, keepdim=True).expand_as(cached_latent).contiguous()
                        if freeze_tactile:
                            # Control for the reactive decoder: hold the tactile history from the refresh
                            # and pad by repeating its last frame, exactly like the open-loop candidate
                            # decode. This makes execution an open-loop chunk (what ViTaL / DICE-RL / LPB do).
                            if int(req.get('chunk_step', 0)) == 0:
                                frozen_ext = {kk: vv.clone() for kk, vv in extended_obs_dict.items()}
                                frozen_age = 0
                            elif frozen_ext is not None:
                                frozen_age += 1
                            if frozen_ext is not None:
                                extended_obs_dict = {kk: torch.cat([vv, vv[:, -1:].expand(-1, frozen_age, -1)], dim=1)
                                                     if frozen_age > 0 else vv for kk, vv in frozen_ext.items()}
                                n_ext = next(iter(extended_obs_dict.values())).shape[-2]
                        if req.get('use_pregen') and PREGEN['c'] is not None and int(req.get('chunk_step', 0)) == 0: PREGEN['used'] += 1
                        if req.get('pregen'):   # rt_final: draw the next refresh's candidates now, from this observation
                            _ne = next(iter(obs_dict.values())).shape[0]; _b = int(constants['dataset_obs_steps'])
                            _tl = lambda d, m: {kk: vv.repeat(m, *([1] * (vv.dim() - 1))) for kk, vv in d.items()}
                            _eb = {kk: vv[:, -_b:] for kk, vv in extended_obs_dict.items()}
                            # rt_x2 AMEND2 (2026-11-14): same row limit as the single-call draw (RDP_DRAW_SINGLE_ROWS, default 128); the 400-row call ran out of GPU memory at 25 envs
                            _mr = max(1, int(__import__('os').environ.get('RDP_DRAW_SINGLE_ROWS', '128')) // _ne); _pp = []; _k = 0
                            while _k < select_k:
                                _m = min(_mr, select_k - _k)
                                _pp.append(policy.predict_action(_tl(obs_dict, _m), dataset_obs_temporal_downsample_ratio=constants['obs_temporal_downsample_ratio'],
                                                                 extended_obs_dict=_tl(_eb, _m), return_latent_action=True)['action'][:, 0, :].view(_m, _ne, -1)); _k += _m
                            PREGEN['c'] = torch.cat(_pp, 0).detach(); PREGEN['made'] += 1
                        if dly_old is not None and int(req.get('chunk_step', 0)) == 0:
                            dly_new = (cached_latent, cached_act_offset); cached_latent, cached_act_offset = dly_old; dly_old = None
                            extended_obs_dict = dly_ext; dly_ext = None; n_ext = next(iter(extended_obs_dict.values())).shape[-2]
                        elif dly_new is not None and req.get('delay_swap'):
                            cached_latent, cached_act_offset = dly_new; dly_new = None
                        result = policy.predict_from_latent_action(
                            cached_latent, extended_obs_dict, n_ext,
                            constants['obs_temporal_downsample_ratio'])
                        # upstream takes the last row of the decoded sequence
                        # (real_runner.py: tcp_step_action = tcp_step_action[-1]);
                        # with a window of 4+i frames that row is the action for now.
                        action = result['action'][:, -1:]
                        if cached_act_offset is not None:
                            action = action + cached_act_offset[:, None].to(action.dtype)
                        recent_actions.append(action[:, 0].detach())
                        recent12.append(action[:, 0].detach())
                        if 'state' in obs_dict:
                            z_history.append(obs_dict['state'][:, -1, 2].detach().clone())
                            pose_history.append(obs_dict['state'][:, -1, :3].detach().clone())
                        step_counter += 1
                        send_msg(conn, {'action': action.detach().cpu().numpy(),
                                        'latent': cached_latent.detach().cpu().numpy(), **({'obs_features': step_feats} if step_feats is not None else {}), **({'evid_done': EVID_STATE['done'].cpu().numpy()} if 'done' in EVID_STATE else {})})
                        continue
                    two_stage = (latent_mode != 'normal' or mode == 'real')
                    if not two_stage:
                        result = policy.predict_action(
                            obs_dict,
                            dataset_obs_temporal_downsample_ratio=constants['obs_temporal_downsample_ratio'],
                            extended_obs_dict=extended_obs_dict,
                            return_latent_action=False)
                    else:
                        # Ablation: does the diffusion's latent carry anything the
                        # rollout depends on? An offline probe put its latent-space
                        # R^2 at -0.01..-0.09, i.e. no better than predicting the mean
                        # latent, which raises the possibility that the policy is
                        # running almost entirely off the AT decoder's extended_obs
                        # conditioning. Replacing the latent per step answers that
                        # directly: 'shuffle' hands each env another env's latent,
                        # 'batchmean' removes per-state variation altogether. If the
                        # rollout barely moves, the diffusion is not contributing.
                        #
                        # Routed through the two-stage API with
                        # extended_obs_last_step = original_horizon and no pad, which
                        # makes get_temporal_cond see exactly what the single-shot
                        # path passes it -- so 'normal' vs these differ only in latent.
                        lat_res = policy.predict_action(
                            obs_dict,
                            dataset_obs_temporal_downsample_ratio=constants['obs_temporal_downsample_ratio'],
                            extended_obs_dict=extended_obs_dict,
                            return_latent_action=True)
                        latent = lat_res['action'][:, 0, :]          # (N, D)
                        if latent_mode == 'shuffle':
                            perm = torch.randperm(latent.shape[0], device=latent.device)
                            latent = latent[perm]
                        elif latent_mode == 'batchmean':
                            latent = latent.mean(dim=0, keepdim=True).expand_as(latent).contiguous()
                        # however many frames the client actually sent; with
                        # extended-obs-mode 'real' that is the real history only,
                        # and get_temporal_cond then does no padding at all.
                        n_ext = next(iter(extended_obs_dict.values())).shape[-2]
                        result = policy.predict_from_latent_action(
                            latent, extended_obs_dict, n_ext,
                            constants['obs_temporal_downsample_ratio'])
                action = result['action'][:, :constants['n_action_steps_exec']]
                send_msg(conn, {'action': action.detach().cpu().numpy(), **({'obs_features': step_feats} if step_feats is not None else {}), **({'evid_done': EVID_STATE['done'].cpu().numpy()} if 'done' in EVID_STATE else {})})
        except ConnectionError:
            print("[policy_server] client disconnected", file=sys.stderr)
        except Exception:
            traceback.print_exc()
            try:
                send_msg(conn, {'error': traceback.format_exc()})
            except Exception:
                pass
        finally:
            if select_rule != 'none':
                print('[policy_server] selection stats: ' + json.dumps(select_stats), file=sys.stderr, flush=True)
            if steer is not None:
                print('[policy_server] steer identity: ' + json.dumps({'checked': steer['n_identity_checked'],
                      'failed': steer['n_identity_failed'], 'scale': steer['scale']}), file=sys.stderr, flush=True)
            if record_path is not None and records:
                np.savez_compressed(record_path, **{kk: np.stack(vv) for kk, vv in records.items()})
                print('[policy_server] wrote ' + str(record_path), file=sys.stderr, flush=True)
            conn.close()


def load_verifier(verifier_ckpt, device, tactile_offset=False):
    """Load --verifier-ckpt (comma list = pessimistic ensemble). A checkpoint with feature_set 'vretrain_v1' loads
    only with tactile_offset=True (--verifier-tactile-offset), and an old checkpoint only without it."""
    from verifier_model import Verifier
    paths = [q for q in verifier_ckpt.split(',') if q]
    members, ck0 = [], None
    for q in paths:
        ck = torch.load(q, map_location=device, weights_only=False)
        m = Verifier(dp=ck.get('dp', 21), dc=ck.get('cand_features', 12), dt=ck.get('dt', 0),
                     use_align=ck.get('use_align', True), use_vision=ck.get('use_vision', True)).to(device)
        m.generic = bool(ck.get('generic', False)); m.rawplan = bool(ck.get('rawplan', False))
        m.feature_set = ck.get('feature_set')
        if ck0 is None:
            ck0 = ck
            if (m.feature_set == VR_FEATURE_SET) != bool(tactile_offset):
                raise ValueError(f"--verifier-tactile-offset must be given iff the checkpoint's feature_set is "
                                 f"{VR_FEATURE_SET} (got {m.feature_set})")
        elif m.feature_set != members[0].feature_set:
            raise ValueError('verifier ensemble members must share feature_set')
        m.load_state_dict(ck['state_dict']); m.eval(); members.append(m)
    stats = {k: (torch.as_tensor(a, device=device, dtype=torch.float32),
                 torch.as_tensor(b, device=device, dtype=torch.float32)) for k, (a, b) in ck0['stats'].items()}
    return {'model': members[0], 'members': members, 'stats': stats, 'tau': float(ck0['tau'])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ldp-ckpt', required=True)
    parser.add_argument('--at-load-dir', required=True)
    parser.add_argument('--task', default='manifeel_pih_c8d_ldp',
                         help='task config name, e.g. manifeel_pih_c8d_ldp or '
                              'manifeel_pih_c8d_ldp_rawtactile')
    parser.add_argument('--at-config', default='at_manifeel_pih',
                         help='AT config the LDP was trained against '
                              '(at_manifeel_pih | at_manifeel_pih_l16)')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--guide-rmax', type=float, default=0.5)
    parser.add_argument('--guide-tau-ref', type=float, default=0.05)
    parser.add_argument('--guide-steps', type=int, default=4)
    parser.add_argument('--guide-cap', type=float, default=6.0)
    parser.add_argument('--guide-force', action='store_true')
    parser.add_argument('--guide-measure-only', action='store_true')
    parser.add_argument('--guide-verifier', default=None, help='pickle {model: sklearn regressor of G on [x 62, c 11]} (train_verifier.py)')
    parser.add_argument('--extended-obs-mode', default='growing',
                        choices=['padded', 'real', 'growing'],
                        help="'padded': client sends original_horizon frames with the future "
                             "filled by repeating the current one (what this bridge has always "
                             "done). 'real': client sends only the frames it actually has, and "
                             "get_temporal_cond is told how many -- which is what upstream's "
                             "RealRunner does (it never passes extend_obs_pad_after). "
                             "rollout_gap_probe puts padded at 2-3x the action MSE of oracle "
                             "extended_obs, near the constant-prediction baseline. "
                             "'growing' additionally reproduces upstream's cadence: the "
                             "diffusion runs once per chunk (chunk_step 0) and the cached "
                             "latent is re-decoded every step against an extended_obs window "
                             "that lengthens as real frames arrive -- real_runner tags each "
                             "row of the chunk with arange(4, 4+len) for exactly this. "
                             "Default 'growing' because that is what upstream does; 'padded' "
                             "is kept only to reproduce this bridge's old behaviour.")
    parser.add_argument('--latent-mode', default='normal',
                        choices=['normal', 'shuffle', 'batchmean'],
                        help="ablation on the diffusion latent; see the serve loop")
    parser.add_argument('--local-radius', type=float, default=0., help='Relative latent perturbation radius; half fresh, half local; learned_q only')
    parser.add_argument('--head-keep-ldp', type=int, default=0, help='Keep this many original LDP candidates before head samples')
    parser.add_argument('--latent-head-ckpt', default=None, help='sample candidate latents from a distilled conditional head instead of the LDP')
    parser.add_argument('--select-pool', type=int, default=0, help='draw this many latents and keep the k most mutually distant before decoding (0 = off)')
    parser.add_argument('--select-every', type=int, default=1, help='run selection only every Nth refresh; other refreshes draw a single latent')
    parser.add_argument('--freeze-tactile', action='store_true', help='execute open-loop within a refresh window: hold the tactile history from the refresh')
    parser.add_argument('--refine-steps', type=int, default=5)
    parser.add_argument('--refine-radius', type=float, default=0.25)
    parser.add_argument('--residual-ckpt', default=None, help='latent residual checkpoint for --select-rule residual')
    parser.add_argument('--record-path', default=None, help='npz path for per-refresh selection records (features, candidates)')
    parser.add_argument('--evid-est', default=None, help='npz alignment estimator enabling the evidence-promoted switch (evidence_rule_20261110)')
    parser.add_argument('--evid-k', type=int, default=2)
    parser.add_argument('--evid-theta', type=float, default=1.0)
    parser.add_argument('--evid-suppress', action='store_true')
    parser.add_argument('--evid-fallback', type=int, default=99)
    parser.add_argument('--evid-perm-k', type=int, default=-1, help='belief_parts_20261116: evidence count c that lifts the suppression (default -1 = same as --evid-k, i.e. unchanged behaviour)')
    parser.add_argument('--coh-lambda', type=float, default=0.0)
    parser.add_argument('--coh-steps', type=int, default=4)
    parser.add_argument('--coh-delay', type=int, default=0)
    parser.add_argument('--evid-soft', type=float, default=0.0, help='soft_belief_20261113: lambda; with --evid-suppress the choice is argmax(Q - lambda * (1 - c/k)+ * dist) while switching is not permitted (0 = hard mask as before)')
    parser.add_argument('--evid-force-cs', type=int, default=10 ** 9)
    parser.add_argument('--evid-ext', action='store_true')
    parser.add_argument('--select-rule', default='none',
                        choices=['none', 'first', 'consistent', 'inconsistent', 'random', 'max_magnitude', 'inconsistent_when_stuck',
                                 'max_magnitude_when_stuck', 'oracle_direction_when_stuck', 'anti_direction_when_stuck', 'oracle_direction',
                                 'oracle_phase', 'learned_q', 'learned_vq', 'residual', 'refine', 'perturb', 'oracle_land', 'explore_q', 'rand_switch', 'term_switch', 'term_guide', 'stop_first', 'mode_first', 'soft_tilt', 'soft_rimorc', 'soft_explore', 'haz', 'haz_soft', 'dec_soft',
                                 'oracle_endpoint', 'dwell_rule'],
                        help="per-refresh selection among --select-k diffusion samples (growing mode only)")
    parser.add_argument('--verifier-ckpt', default=None, help='verifier_final*.pt for learned_q / learned_vq')
    parser.add_argument('--verifier-tau', type=float, default=None,
                        help='override the checkpoint tau for learned_vq: intervene where sigmoid(V) < tau. '
                             '0 never intervenes (= first), 1 always (= learned_q); sweeping it traces success '
                             'and regressions against intervention rate.')
    parser.add_argument('--select-k', type=int, default=8)
    parser.add_argument('--select-horizon', type=int, default=8)
    parser.add_argument('--local-keep', type=int, default=0,
                        help='with --local-radius: number of independent draws to keep, the rest are perturbations '
                             'cycled over them (0 = half/half; 1 = one draw + k-1 perturbations)')
    parser.add_argument('--two-stage-keep', type=int, default=0,
                        help='learned_q only: stage 1 draws this many latents and picks by Q; stage 2 scores the winner '
                             'plus k-1 isotropic perturbations of it (--two-stage-radius x ||z||) and picks again. 0 = off')
    parser.add_argument('--two-stage-radius', type=float, default=0.1)
    parser.add_argument('--local-cov', default=None,
                        help='npz with Vw (D,D) and ew (D,) from within-state covariance of logged candidates; '
                             'perturbations are drawn along the top --local-cov-rank PCs with their own spread')
    parser.add_argument('--local-cov-rank', type=int, default=32)
    parser.add_argument('--codebook', default=None,
                        help='npy (k, D) of coarse latent codes; candidates become the primary draw plus every '
                             'code carrying the primary draw\'s residual (one diffusion call per refresh)')
    parser.add_argument('--draw-chunk', type=int, default=4,
                        help='candidate draws per diffusion call after candidate 0 (1 = original sequential loop)')
    parser.add_argument('--xgrid-clamp-chunk', type=int, default=16,
                        help='latent_clamp: clamped (draw, grid) jobs per diffusion call (each job = n_env rows)')
    parser.add_argument('--xgrid-mode', default='none', choices=['none', 'latent_axis', 'action', 'latent_clamp'],
                        help='opt-in candidate generation (structured-AT study 2026-10-05): draw0 x a fixed grid of '
                             'executed-window x offsets + (draws-1) unshifted draws; see select_latent')
    parser.add_argument('--xgrid-mm', default='0,-4,4,-8,8,-12,12',
                        help='grid of executed-window x offsets, nominal mm (sum of action x over the window x 10); first must be 0')
    parser.add_argument('--xgrid-axis-file', default=None,
                        help="latent_axis: npz with w_unit (D,) and mm_per_latent_unit_exec12 (nominal mm per latent unit)")
    parser.add_argument('--xgrid-draws', type=int, default=2)
    parser.add_argument('--xgrid-all-draws', action='store_true',
                        help='apply the grid to every independent draw (select_k = len(grid) * draws); default: draw0 only')
    parser.add_argument('--xgrid-exec-steps', type=int, default=12, help='executed window length (= bridge latent_refresh)')
    parser.add_argument('--xgrid-gate', default='none', choices=['none', 'tactile_stall', 'external_sign'],
                        help='opt-in (pih_tactile_gate_20261006): learned_q may pick a nonzero-offset candidate only when '
                             'tactile contact (c8d) AND the default stall rule fire; otherwise argmax over offset-0 candidates')
    parser.add_argument('--xgrid-gate-thr', type=float, default=0.02,
                        help='contact threshold on ||c8d[:3] - first-5-step baseline|| (demo free-space p99.9 = 0.018)')
    parser.add_argument('--xgrid-rho', type=float, default=0.4, help='realized/nominal ratio, records only (predicted landing)')
    parser.add_argument('--verifier-obs-encoder-ckpt', default=None,
                        help='compute verifier/record obs_features with THIS LDP checkpoint\'s obs normalizer + obs_encoder '
                             '(EMA) instead of the served policy\'s (opt-in; default unchanged)')
    parser.add_argument('--verifier-tactile-offset', action='store_true',
                        help='opt-in (pih_verifier_retrain_20261006): accept a retrained verifier checkpoint with '
                             "feature_set 'vretrain_v1' (tactile c8d summary T + candidate offset features). Required "
                             'to load such a checkpoint; old checkpoints are rejected with it')
    parser.add_argument('--record-verifier-inputs', action='store_true',
                        help='opt-in: also record per refresh the retrain inputs vr_P, vr_T (tactile), vr_cand (19 '
                             'candidate features), vr_valid, vr_explored, vr_explore_offset_mm (needs --record-path)')
    parser.add_argument('--explore-eps', type=float, default=None,
                        help="--select-rule explore_q: with this prob per env/refresh a grid offset value uniformly "
                             "(incl. 0), else offset 0; the draw is the verifier's argmax Q among that offset")
    parser.add_argument('--explore-seed', type=int, default=0, help='explore_q numpy seed (combined with policy seed)')
    parser.add_argument('--draw-noise', default='global', choices=['global', 'per_draw_seeded'],
                        help='per_draw_seeded (opt-in, budget-compare study): draw d x_T from its own generator seeded by '
                             '(request policy_seed, d); first --draw-noise-base draws bitwise budget-independent')
    parser.add_argument('--draw-noise-base', type=int, default=8)
    parser.add_argument('--record-timing', action='store_true',
                        help='opt-in: CUDA-synchronized per-refresh timing (features/diffusion/decode/score) in the record')
    parser.add_argument('--cache-diffusion-cond', action='store_true',
                        help='opt-in (needs --draw-noise per_draw_seeded): encode the diffusion conditioning once per '
                             'refresh request and reuse it for every draw block; never across requests, tactile not cached')
    parser.add_argument('--steer-mode', default='none', choices=['none', 'latent', 'action'],
                        help='opt-in (latent_steer_20261012): bounded edit after learned_q selection; latent = z0 + PCA-basis '
                             'delta, action = per-window residual added after the reactive AT decode')
    parser.add_argument('--steer-source', default='explore', choices=['explore', 'actor'])
    parser.add_argument('--steer-scale', type=float, default=1.0, help='0 = identity (checked bitwise per refresh)')
    parser.add_argument('--steer-basis', default=None, help='npz with U, sig, A (latent) and b_action (action)')
    parser.add_argument('--steer-actor-ckpt', default=None)
    parser.add_argument('--steer-seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mode-space', default=None)
    parser.add_argument('--haz-curve', default=None, help='hazard_curve.json (multisocket_demo_stall_hazard_20261022)')
    parser.add_argument('--dec-dir', default=None, help='multisocket_dec_sim_20261023/deploy (dec_model.npz, dyn_*.pt, v_obs.pkl)')
    parser.add_argument('--dec-curve', default=None, help='hazard_curve_generic.json (multisocket_generic_stall_offline_20261022)')
    parser.add_argument('--dec-eps', type=float, default=0.5, help='KL budget (nats) of the verifier tilt per refresh')
    parser.add_argument('--no-clip-sample', action='store_true',
                        help='DDIM clip_sample false at inference (usb_keyed_s5c 2026-11-02: latents outside the inherited normalizer range were cut off)')
    parser.add_argument('--record-plans', action='store_true',
                        help='also record the raw decoded select_horizon-step plans of all k candidates (diagnostics only)')
    args = parser.parse_args()
    COH.update(lam=float(args.coh_lambda), m=int(args.coh_steps), d=int(args.coh_delay))
    assert COH['R'] + COH['d'] + COH['m'] <= 24, 'continuity window exceeds the decoded plan'
    GUIDE.update(rmax=args.guide_rmax, tau_ref=args.guide_tau_ref, steps=args.guide_steps, cap=args.guide_cap,
                 force=bool(args.guide_force), measure_only=bool(args.guide_measure_only))
    if args.guide_verifier:
        import pickle
        with open(args.guide_verifier, 'rb') as fh:
            GUIDE['verifier'] = pickle.load(fh)['model']
    print('[policy_server] GUIDE', GUIDE, file=sys.stderr)
    if args.mode_space:
        _z = np.load(args.mode_space)
        MODE_SPACE.update({kk: np.asarray(_z[kk], np.float64) for kk in ('sd', 'pca_mean', 'pca_components', 'centers')})
        MODE_SPACE.update({kk: np.asarray(_z[kk], np.float64) for kk in ('share', 'sigma2', 'eps0', 'lmax') if kk in _z.files})
        print('[policy_server] MODE_SPACE k', MODE_SPACE['centers'].shape[0], file=sys.stderr)
    if args.dec_dir:
        load_dec(args.dec_dir, args.dec_curve, args.dec_eps, args.device if hasattr(args, 'device') else 'cuda')
    if args.haz_curve:
        _hc = json.loads(open(args.haz_curve).read())
        HAZ.update(a=float(_hc['a']), b=float(_hc['b']), d_cap=int(_hc['d_cap']))
        print('[policy_server] HAZ', HAZ, file=sys.stderr)
    if args.cache_diffusion_cond and args.draw_noise != 'per_draw_seeded':
        parser.error('--cache-diffusion-cond needs --draw-noise per_draw_seeded')

    policy, constants = load_policy(args.ldp_ckpt, args.at_load_dir, args.task,
                                    args.device, args.at_config,
                                    extended_obs_mode=args.extended_obs_mode)
    if args.no_clip_sample:
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        policy.noise_scheduler = DDIMScheduler.from_config({**policy.noise_scheduler.config, 'clip_sample': False})
        constants['clip_sample'] = False
        print('[policy_server] DDIM clip_sample disabled (--no-clip-sample)', file=sys.stderr)
    verifier = None
    if args.evid_est:
        global EVID
        _e = np.load(args.evid_est); _dv = args.device if hasattr(args, 'device') else 'cuda'; _k = torch.as_tensor(_e['k'], device=_dv)
        _t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=_dv)
        EVID = {'k': _k, 'mu': _t(_e['mu'])[_k], 'sd': _t(_e['sd'])[_k], 'wx': _t(_e['wx']), 'bx': float(_e['bx']), 'wr': _t(_e['wr']), 'br': float(_e['br']),
                'kstar': int(args.evid_k), 'perm_k': (int(args.evid_perm_k) if int(args.evid_perm_k) >= 0 else int(args.evid_k)), 'theta': float(args.evid_theta), 'suppress': bool(args.evid_suppress), 'fallback': int(args.evid_fallback), 'soft': float(args.evid_soft), 'force_cs': int(args.evid_force_cs), 'ext': bool(args.evid_ext)}
        print('evidence rule enabled', int(_k.sum()), EVID['kstar'], EVID['theta'], flush=True)
    if args.verifier_ckpt:
        try:
            verifier = load_verifier(args.verifier_ckpt, args.device, args.verifier_tactile_offset)
        except ValueError as e:
            parser.error(str(e))
        if args.verifier_tau is not None:
            verifier['tau'] = float(args.verifier_tau)
        if len(verifier["members"]) > 1:
            print(f"[policy_server] verifier ensemble of {len(verifier['members'])} (pessimistic min over A)", file=sys.stderr)
        print(f"[policy_server] loaded verifier {args.verifier_ckpt} tau={verifier['tau']:.3f}", file=sys.stderr)
    if args.select_rule in ('learned_q', 'learned_vq', 'explore_q') and verifier is None:
        parser.error('learned rules need --verifier-ckpt')
    explore = None
    if args.select_rule == 'explore_q':
        if args.explore_eps is None or not 0.0 <= args.explore_eps <= 1.0:
            parser.error('explore_q needs --explore-eps in [0, 1]')
        if args.xgrid_mode == 'none' or not args.xgrid_all_draws:
            parser.error('explore_q needs --xgrid-mode with --xgrid-all-draws')
        explore = {'eps': float(args.explore_eps), 'seed': int(args.explore_seed)}
    if args.record_verifier_inputs and (args.record_path is None or args.xgrid_mode == 'none'):
        parser.error('--record-verifier-inputs needs --record-path and --xgrid-mode')
    if args.record_timing and args.record_path is None:
        parser.error('--record-timing needs --record-path')
    if args.draw_noise != 'global' and (args.xgrid_mode == 'latent_clamp' or args.latent_head_ckpt or args.local_radius > 0
                                        or args.codebook or args.select_pool > 0 or args.two_stage_keep > 0):
        parser.error('--draw-noise per_draw_seeded supports plain draws and latent_axis/action xgrid only')
    need_tactile = bool(args.record_verifier_inputs) or (verifier is not None and verifier['model'].feature_set == VR_FEATURE_SET)
    residual = None
    if args.residual_ckpt:
        from latent_residual import LatentResidual
        rk = torch.load(args.residual_ckpt, map_location=args.device, weights_only=False)
        rm = LatentResidual(dv=rk['dv'], dp=rk['dp'], dz=rk['dz'], scale=rk.get('scale', 1.0)).to(args.device)
        rm.load_state_dict(rk['state_dict']); rm.eval()
        rstats = {k: (torch.as_tensor(a, device=args.device, dtype=torch.float32),
                      torch.as_tensor(b, device=args.device, dtype=torch.float32)) for k, (a, b) in rk['stats'].items()}
        residual = {'model': rm, 'stats': rstats}
        print(f"[policy_server] loaded latent residual {args.residual_ckpt} scale={rk.get('scale', 1.0)}", file=sys.stderr)
    if args.select_rule == 'residual' and residual is None:
        parser.error('--select-rule residual needs --residual-ckpt')
    head = None
    if args.latent_head_ckpt:
        from latent_head import LatentHead
        hk = torch.load(args.latent_head_ckpt, map_location=args.device, weights_only=False)
        hm = LatentHead(dv=hk['dv'], dz=hk['dz']).to(args.device); hm.load_state_dict(hk['state_dict']); hm.eval()
        hst = {k: (torch.as_tensor(a, device=args.device, dtype=torch.float32),
                   torch.as_tensor(b, device=args.device, dtype=torch.float32)) for k, (a, b) in hk['stats'].items()}
        head = {'model': hm, 'stats': hst, 'keep_ldp': args.head_keep_ldp, 'conditioning': hk.get('conditioning', 'vision')}
        print(f"[policy_server] loaded latent head {args.latent_head_ckpt} (rows={hk.get('rows')})", file=sys.stderr)
    if args.select_rule == 'refine' and verifier is None:
        parser.error('--select-rule refine needs --verifier-ckpt')
    local_cov = None
    if args.local_cov:
        cz = np.load(args.local_cov)
        m = int(args.local_cov_rank)
        Lm = cz['Vw'][:, :m] * np.sqrt(np.clip(cz['ew'][:m], 0, None))[None]
        local_cov = torch.as_tensor(Lm, dtype=torch.float32, device=args.device)
        print(f"[policy_server] shaped local perturbation: rank {m}, natural RMS {float(np.sqrt(cz['ew'][:m].sum())):.3f}", file=sys.stderr)
    codebook = None
    if args.codebook:
        codebook = torch.as_tensor(np.load(args.codebook), dtype=torch.float32, device=args.device)
        print(f"[policy_server] codebook {tuple(codebook.shape)} from {args.codebook}", file=sys.stderr)
    xgrid = None
    if args.xgrid_gate != 'none' and args.xgrid_mode == 'none':
        parser.error('--xgrid-gate needs --xgrid-mode')
    if args.xgrid_mode != 'none':
        if args.select_rule in ('residual', 'refine', 'perturb') or args.two_stage_keep > 0:
            parser.error('xgrid is a candidate generator for selecting rules only')
        xgrid = {'mode': args.xgrid_mode, 'mm': [float(v) for v in args.xgrid_mm.split(',')], 'draws': args.xgrid_draws,
                 'all_draws': bool(args.xgrid_all_draws),
                 'exec_steps': args.xgrid_exec_steps, 'rho': args.xgrid_rho,
                 'action_dim': int(policy.normalizer['action'].get_input_stats()['mean'].numel())}
        if args.xgrid_gate != 'none':
            if args.select_rule not in ('learned_q', 'learned_vq'):
                parser.error('--xgrid-gate needs --select-rule learned_q / learned_vq')
            if args.xgrid_gate == 'external_sign' and (args.select_rule != 'learned_q' or args.two_stage_keep > 0):
                parser.error('--xgrid-gate external_sign supports --select-rule learned_q without two-stage only')
            xgrid['gate'] = args.xgrid_gate
            xgrid['gate_thr'] = args.xgrid_gate_thr
        if args.xgrid_mode == 'latent_clamp':
            if not args.xgrid_all_draws:
                parser.error('--xgrid-mode latent_clamp needs --xgrid-all-draws')
            xgrid['clamp_chunk'] = args.xgrid_clamp_chunk
        if args.xgrid_mode in ('latent_axis', 'latent_clamp'):
            if not args.xgrid_axis_file:
                parser.error('--xgrid-mode latent_axis/latent_clamp needs --xgrid-axis-file')
            ax = np.load(args.xgrid_axis_file)
            xgrid['w'] = torch.as_tensor(ax['w_unit'], dtype=torch.float32, device=args.device)
            xgrid['mm_per_unit'] = float(ax['mm_per_latent_unit_exec12'])
        print(f"[policy_server] xgrid {json.dumps({kk: vv for kk, vv in xgrid.items() if kk != 'w'})}", file=sys.stderr)
    feat_encoder = None
    if args.verifier_obs_encoder_ckpt:
        feat_encoder = load_feature_encoder(args.verifier_obs_encoder_ckpt, args.device, args.at_config)
        constants['verifier_obs_encoder_ckpt'] = feat_encoder['ckpt']
        print(f"[policy_server] verifier obs_features from {feat_encoder['ckpt']}", file=sys.stderr)
    steer = None
    if args.steer_mode != 'none':
        if args.select_rule != 'learned_q' or args.xgrid_mode != 'none' or args.two_stage_keep > 0 or not args.steer_basis:
            parser.error('--steer-mode needs plain --select-rule learned_q (no xgrid / two-stage) and --steer-basis')
        if (args.steer_source == 'actor') != bool(args.steer_actor_ckpt):
            parser.error('--steer-actor-ckpt is required iff --steer-source actor')
        from latent_steer import load_steer
        steer = load_steer(args.steer_mode, args.steer_source, args.steer_scale, args.steer_basis, args.steer_actor_ckpt,
                           args.steer_seed, args.device, int(policy.normalizer['action'].get_input_stats()['mean'].numel()))
        need_tactile = True
        print(f"[policy_server] steer mode={args.steer_mode} source={args.steer_source} scale={args.steer_scale} m={steer['m']}",
              file=sys.stderr)
    global RECORD_PLANS
    RECORD_PLANS = bool(args.record_plans)
    serve(policy, constants, args.port, args.device, args.latent_mode,
          args.select_rule, args.select_k, args.select_horizon, args.record_path, verifier, residual,
          args.refine_steps, args.refine_radius, args.freeze_tactile, args.select_every, args.select_pool, head, args.local_radius,
          args.draw_chunk, codebook, args.local_keep, local_cov, args.two_stage_keep, args.two_stage_radius, xgrid,
          feat_encoder, explore, bool(args.record_verifier_inputs), need_tactile,
          args.draw_noise, args.draw_noise_base, bool(args.record_timing), cache_cond=bool(args.cache_diffusion_cond), steer=steer)


if __name__ == '__main__':
    main()
