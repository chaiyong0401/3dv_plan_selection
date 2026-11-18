"""eval_rdp_bridge_usb_keyed_conv_20261111.py (new 2026-11-11; link_cl_20261111) = eval_rdp_bridge_usb_keyed_handframe_20261106.py with (1) the residual class
DeltaResidualHFC (conv_cls network: correction + convergence head), (2) --evid_theta: the head margins of the previous step are sent to the server as evid_ext.
eval_rdp_bridge_usb_keyed_handframe_20261106.py (new 2026-11-06; usb_keyed_s12u): copy of eval_rdp_bridge_usb_keyed_gatefree_20261106.py with
ONE change: --delta_mode deploy_gf loads delta_residual_handframe_20261106.DeltaResidualHF (hand-frame wrapper). Docstring follows.
eval_rdp_bridge_usb_keyed_gatefree_20261106.py (new 2026-11-06; usb_keyed_s12r): copy of eval_rdp_bridge_usb_keyed_res_20261102.py
with extra --delta_mode collect_gf | deploy_gf using delta_residual_gatefree_20261106.DeltaResidualGF (no contact gate; privileged
3-D plug/socket poses passed for the teacher zone and logging). Modes off / collect / deploy unchanged. Original docstring follows.
Keyed-USB residual bridge (new file, 2026-11-02, usb_keyed_s6_res_20261102) = eval_rdp_bridge_usb_keyed_20261028.py plus
--delta_mode {off, collect, deploy} / --delta_ckpt with delta_residual_ms_rld_20261024.DeltaResidual (the multisocket CR-DAgger-style
residual, unchanged): contact gate -Fz > 1 N (hand frame, wrist wrench), inputs low-dim 100 + the served policy's per-step
obs_features 2078 (requests carry want_feats=True; the server must be policy_server_selrule_decu_20261024.py) + raw right force
field 840. collect = the privileged expert's bounded delta (xy toward the socket centre, orientation toward the NEARER of the two
180-deg-symmetric orientations, |delta| <= 0.2) is EXECUTED (round 1); deploy = the trained residual. The socket is passed as a single
slot (slot_xy (n, 1, 2), slot_quat (n, 1, 4)); the expert never decides a flip. Writes delta_log.npz; extra traj keys action_policy /
action_exec.

Base docstring: Keyed-USB evaluation bridge (new file, 2026-10-28) = eval_rdp_bridge_usb_20261027.py plus two options for stage 1d:
  * --cfg_overrides JSON: dotted isaacgym_config overrides applied before the env is built (stage 1d passes the symmetric
    visual asset: task.env.asset_info_filename tacsl_asset_info_usb_symvis_20261028.yaml, desired_subassemblies
    [USB_plug_symvis], as usb_keyed_s0c..s1b).
  * --socket_flip: per-env socket yaw 0 / pi, Bernoulli(0.5) from np.random.default_rng(test_start_seed + 7) (= the s1a/s1b
    teacher rule), applied inside task._reset_object on every reset.  The flip vector is logged (initial_state.npz,
    eval_log.json); it is never sent to the policy.  Extra logged traj key: socket_xaxis.
Everything else (wire protocol, obs pipeline, refresh, outputs) is unchanged from the 20261027 bridge, whose docstring follows.

USB evaluation bridge for RDP (AT + LDP) policies (new file, 2026-10-27).

= eval_rdp_bridge_multisocket_20261018.py (same wire protocol, obs pipeline, 'growing' extended-obs windowing, global
latent refresh, refresh_check) with the multi-socket parts removed:
  * env: TacSLTaskUSB (manifeel/config/isaacgym_config_usb.yaml) through the base MultipleIsaacEnvWrapper; the class is
    replaced in this process by usb_wrist_wrench_20261027.install() so the multi-socket wrist F/T estimator runs
    (task.wrist_wrench = the TacSLTaskInsertionMultiSocket.yaml block, noise seed = --wrench_noise_seed, default
    test_start_seed + 7_000_000).  The wrench is logged, never sent.
  * no slot / target_side / distractor; success = task._check_success() (USB keypoint rule), any step.  No strict rule.
  * logged pose keys: plug_pos/quat, socket_pos/quat, ee_pos/quat, p_ih / q_ih (plug pose in the fingertip frame =
    inhand_teacher_20261026.inhand_pose), wrist_wrench(_clean), plug_socket_force (privileged pairwise, logged only).
Policy input unchanged: wrist, front, state (ee_pos + ee_quat), contact_8d_right (live c8d of the right force field).
Outputs as the multi-socket bridge (run_metadata.json, sim_config.yaml, rollout_traj.npz, initial_state.npz,
refresh_log.npz, eval_log.json).  Sim only; robot_command_sent: false.
"""
import isaacgym  # noqa: F401  must be imported before torch anywhere in the process

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import collections
import json
import math
import pathlib
import pickle
import socket
import struct
import time

import click
import hydra
import numpy as np
from omegaconf import OmegaConf

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'scripts'))
import convert_force_field_to_c8d as c8d_lib  # noqa: E402

NOISE_SEED_SALT = 7_000_000          # = multi-socket bridge / recorders


def recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('policy_server closed the connection')
        buf += chunk
    return buf


def recv_msg(sock):
    (length,) = struct.unpack('>I', recv_exact(sock, 4))
    return pickle.loads(recv_exact(sock, length))


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack('>I', len(data)) + data)


class PolicyClient:
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.constants = recv_msg(self.sock)
        self.last_latent = None
        print(f"[usb_keyed_bridge] policy_server constants: {self.constants}")

    def predict(self, obs, extended_obs, chunk_step=0, **options):
        send_msg(self.sock, {'obs': obs, 'extended_obs': extended_obs, 'chunk_step': int(chunk_step), **options})
        resp = recv_msg(self.sock)
        if 'error' in resp:
            raise RuntimeError(f"policy_server error:\n{resp['error']}")
        self.last_latent = resp.get('latent')
        self.last_feats = resp.get('obs_features')
        return resp['action']

    def close(self):
        try:
            send_msg(self.sock, {'cmd': 'close'})
        except Exception:
            pass
        self.sock.close()


class LiveC8DConverter:
    def __init__(self, rows=10, cols=14):
        full_coords = c8d_lib.build_taxel_coords(rows, cols, c8d_lib.TAXEL_SPACING_M).reshape(rows, cols, 3)
        self.pooled_coords = c8d_lib.pool_2x2_mean(full_coords).reshape(-1, 3)

    def __call__(self, force_field_hwc):
        pooled = c8d_lib.pool_2x2_sum(force_field_hwc)
        return c8d_lib.compute_c8d(pooled, self.pooled_coords)


def refresh_schedule(n_steps, R):
    """chunk_step per step and the steps at which the diffusion runs (pure, unit-checked in __main__ --selftest)."""
    cs = np.arange(n_steps) % R
    return cs, np.flatnonzero(cs == 0)


@click.command()
@click.option('-o', '--output_dir', required=True)
@click.option('-n', '--cfg_name', default='isaacgym_config_usb.yaml')
@click.option('--host', default='127.0.0.1')
@click.option('--port', default=8765, type=int)
@click.option('--n_test', default=50, type=int)
@click.option('--n_test_vis', default=0, type=int)
@click.option('--max_steps', default=400, type=int)
@click.option('--stop_after', default=None, type=int, help='shorter diagnostic rollout; task budget stays max_steps')
@click.option('--test_start_seed', default=10000, type=int)
@click.option('--latent_refresh', default=12, type=int, help='global refresh period R (diffusion every R steps)')
@click.option('--allow_window_saturation', is_flag=True)
@click.option('--policy_seed', default=None, type=int, help='seed each policy request by seed + control step')
@click.option('--wrench_noise_seed', default=None, type=int, help='default test_start_seed + 7_000_000')
@click.option('--server_record_path', default=None, help='path the selrule server writes (--record-path); metadata')
@click.option('--cfg_overrides', default='{}', help='JSON {dotted.key: value} applied to the isaacgym config')
@click.option('--socket_flip', is_flag=True, help='per-env socket yaw pi with p 0.5, rng default_rng(test_start_seed + 7)')
@click.option('--delta_mode', type=click.Choice(['off', 'collect', 'deploy', 'collect_gf', 'deploy_gf']), default='off')
@click.option('--delta_ckpt', default=None)
@click.option('--exec_delay', default=0, type=int, help='rt_robust_20261113: a refreshed latent takes effect this many steps late (previous latent continues meanwhile)')
@click.option('--exec_pregen', default=0, type=int, help='rt_final_20261113: candidates for a refresh are drawn this many steps EARLY (request flag pregen); at the refresh the server scores the stored candidates with the current observation (use_pregen); the decode window is longer by this lead so the executed latent starts at that step')
@click.option('--delay_noskip', is_flag=True, help='rt_noskip_20261113: with --exec_delay, the late latent starts at ITS OWN step 0 when it takes over (decode window restarts) instead of entering at step exec_delay')
@click.option('--rate_limit', default=None, help='rt_robust_20261113: Lxyz,Lrot = max step-to-step change of the executed action (norm, normalized units)')
@click.option('--evid_theta', default=None, help='theta_xy,theta_rot: send the residual head margins (logit - theta) of the previous step as evid_ext')
@click.option('--overwrite', is_flag=True)
def main(output_dir, cfg_name, host, port, n_test, n_test_vis, max_steps, stop_after, test_start_seed, latent_refresh,
         allow_window_saturation, policy_seed, wrench_noise_seed, server_record_path,
         cfg_overrides, socket_flip, delta_mode, delta_ckpt, evid_theta, overwrite, exec_delay=0, rate_limit=None, delay_noskip=False, exec_pregen=0):
    if min(n_test, max_steps, latent_refresh) < 1 or not 0 <= n_test_vis <= n_test:
        raise click.BadParameter('invalid n_test / max_steps / latent_refresh / n_test_vis')
    wrench_noise_seed = test_start_seed + NOISE_SEED_SALT if wrench_noise_seed is None else wrench_noise_seed
    horizon_steps = min(max_steps, stop_after or max_steps)
    output_path = pathlib.Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise click.BadParameter(f'{output_dir} exists and is not empty (use --overwrite)')
    output_path.mkdir(parents=True, exist_ok=True)

    client = PolicyClient(host, port)
    c = client.constants
    dataset_obs_steps = c['dataset_obs_steps']
    ratio = c['obs_temporal_downsample_ratio']
    original_horizon = c['original_horizon']
    extended_obs_mode = c.get('extended_obs_mode', 'padded')
    if extended_obs_mode != 'growing':
        raise RuntimeError(f"latent_refresh semantics need the server in 'growing' mode, got {extended_obs_mode}")
    max_window = dataset_obs_steps + latent_refresh - 1
    if max_window > original_horizon and not allow_window_saturation:
        raise RuntimeError(f'R={latent_refresh}: decode window {max_window} > original_horizon {original_horizon}')

    obs_keys, extended_obs_keys = c['obs_keys'], c['extended_obs_keys']
    needed_keys = set(obs_keys) | set(extended_obs_keys)
    known_keys = {'wrist', 'front', 'state', 'contact_8d_right'}   # wrist_wrench is logged, never sent
    unknown = needed_keys - known_keys
    if unknown:
        raise RuntimeError(f'policy_server wants keys {unknown} this bridge does not send (known: {known_keys})')

    hydra.initialize_config_dir(config_dir=str(HERE / 'manifeel' / 'config'), version_base=None)
    task_cfg = hydra.compose(config_name='train_diffusion_workspace.yaml', overrides=['task=visff_wrist'])
    shape_meta = task_cfg.task.shape_meta
    if 'front' in needed_keys and 'front' not in shape_meta['obs']:
        OmegaConf.set_struct(shape_meta, False)
        shape_meta['obs']['front'] = OmegaConf.to_container(shape_meta['obs']['wrist'])
    isaacgym_cfg = hydra.compose(config_name=cfg_name)
    overrides = json.loads(cfg_overrides)
    for key, value in overrides.items():
        node = isaacgym_cfg
        *path, last = key.split('.')
        for p in path:
            node = node[p]
        if last not in node:
            raise click.BadParameter(f'override key {key} not in config')
        node[last] = value
    if isaacgym_cfg.task.name != 'TacSLTaskUSB':
        raise RuntimeError(f'{cfg_name} is not the USB task ({isaacgym_cfg.task.name})')
    action_dim = int(isaacgym_cfg.task.env.numActions)
    shape_meta.action.shape = [action_dim]
    OmegaConf.set_struct(isaacgym_cfg, False)
    isaacgym_cfg['shape_meta'] = shape_meta
    isaacgym_cfg.num_envs = n_test
    isaacgym_cfg.task.rl.max_episode_length = max_steps
    import usb_wrist_wrench_20261027 as uww
    isaacgym_cfg.task['wrist_wrench'] = uww.ms_wrist_wrench_cfg(noise_seed=int(wrench_noise_seed))
    pos_scale = [float(x) for x in isaacgym_cfg.task.rl.pos_action_scale]

    flip = np.random.default_rng(test_start_seed + 7).random(n_test) < 0.5 if socket_flip else np.zeros(n_test, bool)
    metadata = dict(bridge='eval_rdp_bridge_usb_keyed_res_20261102.py', delta_mode=delta_mode, delta_ckpt=delta_ckpt, cfg_name=cfg_name, n_test=n_test,
                    cfg_overrides=overrides, socket_flip=socket_flip,
                    flip_rng='np.random.default_rng(test_start_seed + 7).random(n_test) < 0.5', flip=flip.tolist(),
                    max_steps=max_steps, stop_after=stop_after, test_start_seed=test_start_seed,
                    policy_seed=policy_seed, latent_refresh=latent_refresh,
                    wrench_noise_seed=wrench_noise_seed,
                    server_record_path=server_record_path, policy=c, pos_action_scale=pos_scale,
                    max_decode_window=min(max_window, original_horizon), obs_sent_to_policy=sorted(needed_keys),
                    logged_not_sent=['wrist_wrench', 'plug pose', 'p_ih/q_ih', 'plug_socket_force'],
                    robot_command_sent=False)

    import manifeel.envs.vistac_isaacgym_multiple_env_wrapper as msw
    uww.install(msw.isaacgym_task_map)   # TacSLTaskUSB -> TacSLTaskUSBWrench (wrist F/T), this process only
    import inhand_teacher_20261026 as iht
    from manifeel.gym_util.video_recording_wrapper import VideoRecordingWrapper
    import torch
    from isaacgym import torch_utils
    OmegaConf.save(isaacgym_cfg, output_path / 'sim_config.yaml', resolve=True)
    base_env = msw.MultipleIsaacEnvWrapper(isaacgym_cfg)
    env = VideoRecordingWrapper(base_env, output_dir=output_dir, n_records=n_test_vis, fps=10, crf=22,
                                file_paths=None, steps_per_render=1) if n_test_vis else base_env
    task = base_env.envs
    assert type(task).__name__ == 'TacSLTaskUSBWrench' and getattr(task, 'ww_enabled', False), 'USB task / wrench'
    metadata['socket_asset'] = getattr(task, 'asset_file_paths', {}).get('socket')
    if socket_flip:
        from isaacgym import gymtorch
        flip_t = torch.as_tensor(flip, device=task.device)
        orig_reset_object = task._reset_object

        def reset_object_flip(env_ids):
            orig_reset_object(env_ids)
            ids = torch.as_tensor(env_ids, device=task.device).long()
            ids = ids[flip_t[ids]]
            if len(ids) == 0:
                return
            task.root_quat[ids, task.socket_actor_id_env] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=task.device)  # yaw pi (xyzw)
            sim_ids = task.actor_ids_sim_tensors['socket'][ids]
            task.gym.set_actor_root_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task.root_state),
                                                         gymtorch.unwrap_tensor(sim_ids), len(sim_ids))

        task._reset_object = reset_object_flip

    c8d = LiveC8DConverter()
    n_envs = n_test
    buffers = [collections.deque(maxlen=dataset_obs_steps) for _ in range(n_envs)]
    ext_buffers = [collections.deque(maxlen=original_horizon) for _ in range(n_envs)]

    def frames_from_obs(obs):
        fields = {}
        if 'wrist' in needed_keys:
            fields['wrist'] = obs['wrist'].astype(np.float32)
        if 'front' in needed_keys:
            fields['front'] = obs['front'].astype(np.float32)
        if 'state' in needed_keys:
            fields['state'] = obs['state'].astype(np.float32)
        raw_chw = obs['tactile_force_field_right'].astype(np.float32)
        fields['contact_8d_right'] = c8d(np.moveaxis(raw_chw, 1, -1))   # always computed (logged)
        return fields

    def push_frame(obs, fill=False):
        fields = frames_from_obs(obs)
        send = {k: v for k, v in fields.items() if k in needed_keys}
        for i in range(n_envs):
            frame = {k: v[i] for k, v in send.items()}
            for _ in range(dataset_obs_steps if fill else 1):
                buffers[i].append(frame)
                ext_buffers[i].append(frame)
        return fields

    def build_request(chunk_step, long=False, n_override=None):
        obs_req = {k: [] for k in obs_keys}
        ext_req = {k: [] for k in extended_obs_keys}
        n_want = min(dataset_obs_steps + chunk_step + (latent_refresh if long else 0), original_horizon) if n_override is None else min(int(n_override), original_horizon)
        for i in range(n_envs):
            sel = list(buffers[i])[::-ratio][::-1]
            for k in obs_keys:
                obs_req[k].append(np.stack([f[k] for f in sel]))
            ext_frames = list(ext_buffers[i])[-n_want:]
            for k in extended_obs_keys:
                ext_req[k].append(np.stack([f[k] for f in ext_frames]))
        return {k: np.stack(v) for k, v in obs_req.items()}, {k: np.stack(v) for k, v in ext_req.items()}, n_want

    def _np(v):
        return np.asarray(v.detach().cpu() if hasattr(v, 'detach') else v, dtype=np.float32)

    def pose_state():
        q_ih, p_ih = iht.inhand_pose(task)
        return dict(plug_pos=_np(task.plug_pos), plug_quat=_np(task.plug_quat),
                    socket_pos=_np(task.socket_pos), socket_quat=_np(task.socket_quat),
                    ee_pos=_np(task.fingertip_midpoint_pos), ee_quat=_np(task.fingertip_midpoint_quat),
                    p_ih=_np(p_ih), q_ih=_np(q_ih),
                    wrist_wrench=_np(task.wrist_wrench_meas), wrist_wrench_clean=_np(task.wrist_wrench_clean),
                    plug_socket_force=_np(iht.plug_socket_force(task)),
                    socket_xaxis=_np(torch_utils.quat_rotate(task.socket_quat, torch.tensor(
                        [1.0, 0.0, 0.0], device=task.device).expand(n_envs, 3))))

    env.seed(test_start_seed)
    obs = env.reset()
    push_frame(obs, fill=True)
    init = pose_state()
    metadata.update(ww_noise_seed_used=getattr(task, 'ww_noise_seed_used', None))
    (output_path / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    sx = init['socket_xaxis'][:, 0]
    if socket_flip and not (np.all(sx[flip] < -0.9) and np.all(sx[~flip] > 0.9)) and not (np.all(sx[flip] > 0.9) and np.all(sx[~flip] < -0.9)):
        raise RuntimeError(f'socket flip not applied: x-axis x {np.round(sx, 2)} flip {flip}')
    np.savez_compressed(output_path / 'initial_state.npz', flip=flip, root_state=_np(task.root_state),
                        dof_state=_np(task.dof_state), **{'init_' + k: v for k, v in init.items()})

    traj = collections.defaultdict(list)
    rlog = collections.defaultdict(list)
    done = np.zeros(n_envs, dtype=bool)
    prev_latent = None
    prev_exec = None
    lat_start = 0
    steps_taken = 0
    dres = None
    if delta_mode in ('collect_gf', 'deploy_gf'):
        from delta_residual_errsim_20261118 import DeltaResidualES as DeltaResidualGF
        if delta_mode == 'deploy_gf' and not delta_ckpt:
            raise click.BadParameter('--delta_mode deploy_gf needs --delta_ckpt')
        dres = DeltaResidualGF(n_envs, delta_mode[:-3], delta_ckpt)
        dres.feats_src = client
        ff_init = np.asarray(obs['tactile_force_field_right'], np.float32).copy()
    elif delta_mode != 'off':
        from delta_residual_ms_rld_20261024 import DeltaResidual
        if delta_mode == 'deploy' and not delta_ckpt:
            raise click.BadParameter('--delta_mode deploy needs --delta_ckpt')
        dres = DeltaResidual(n_envs, delta_mode, None, delta_ckpt)
        dres.feats_src = client
        ff_init = np.asarray(obs['tactile_force_field_right'], np.float32).copy()
    pre = init
    while not np.all(done) and steps_taken < horizon_steps:
        chunk_step = steps_taken % latent_refresh
        dly = exec_delay > 0 and steps_taken >= latent_refresh
        if delay_noskip and dly and chunk_step == exec_delay: lat_start = steps_taken
        pg_off = exec_pregen if (exec_pregen > 0 and steps_taken >= latent_refresh) else 0
        obs_req, ext_req, n_ext = build_request(chunk_step, long=dly and chunk_step < exec_delay, n_override=(dataset_obs_steps + steps_taken - lat_start) if (delay_noskip and exec_delay > 0) else ((dataset_obs_steps + chunk_step + pg_off) if exec_pregen > 0 else None))
        options = {} if policy_seed is None else {'policy_seed': policy_seed + steps_taken}
        if dly and chunk_step == 0: options['delay_base_n'] = int(dataset_obs_steps)
        if dly and chunk_step == exec_delay: options['delay_swap'] = True
        if exec_pregen > 0 and chunk_step == latent_refresh - exec_pregen: options['pregen'] = True
        if exec_pregen > 0 and chunk_step == 0 and steps_taken >= latent_refresh: options['use_pregen'] = True
        if dres is not None:
            options['want_feats'] = True
            if evid_theta and getattr(dres, 'head', None) is not None:
                options['evid_ext'] = (dres.head - np.array([float(v) for v in evid_theta.split(',')], np.float32)).astype(np.float32)
        t0 = time.perf_counter()
        action_chunk = client.predict(obs_req, ext_req, chunk_step, **options)
        dt = time.perf_counter() - t0
        if action_chunk.shape[1] != 1:
            raise RuntimeError(f'server returned {action_chunk.shape[1]} actions per request in growing mode; '
                               'the global refresh schedule requires exactly 1')
        action = np.asarray(action_chunk[:, 0], dtype=np.float32)
        if action.shape != (n_envs, action_dim) or not np.isfinite(action).all():
            raise RuntimeError(f'invalid policy action {action.shape}')
        lat = None if client.last_latent is None else np.asarray(client.last_latent, dtype=np.float32).reshape(n_envs, -1)
        changed = (np.ones(n_envs, bool) if prev_latent is None else np.any(lat != prev_latent, axis=-1)) \
            if lat is not None else np.zeros(n_envs, bool)
        prev_latent = lat
        if chunk_step == 0:
            rlog['step'].append(steps_taken)
            rlog['request_seconds'].append(dt)
            for k in ('ee_pos', 'plug_pos', 'p_ih'):
                rlog[k].append(pre[k])
            if lat is not None:
                rlog['latent'].append(lat)

        traj['action_policy'].append(action.copy())
        if dres is not None and delta_mode.endswith('_gf'):
            action = dres.step(action, obs, frames_from_obs(obs)['contact_8d_right'], traj['wrist_wrench'], traj['success'],
                               pre['plug_pos'], pre['socket_pos'][:, None], obs['tactile_force_field_right'], ff_init,
                               pre['plug_quat'], pre['socket_quat'][:, None])
        elif dres is not None:
            action = dres.step(action, obs, frames_from_obs(obs)['contact_8d_right'], traj['wrist_wrench'], traj['success'],
                               pre['plug_pos'][:, :2], pre['socket_pos'][:, None, :2], obs['tactile_force_field_right'], ff_init,
                               pre['plug_quat'], pre['socket_quat'][:, None])
        if rate_limit and prev_exec is not None:
            for sl_, L_ in zip((slice(0, 3), slice(3, 6)), (float(v) for v in rate_limit.split(','))):
                dl_ = action[:, sl_] - prev_exec[:, sl_]; action[:, sl_] = prev_exec[:, sl_] + dl_ * np.minimum(1.0, L_ / np.maximum(np.linalg.norm(dl_, axis=1, keepdims=True), 1e-9))
        prev_exec = np.asarray(action, np.float32).copy()
        traj['action_exec'].append(action.copy())
        obs, reward, reset, info = env.step(action)
        fields = push_frame(obs)
        post = pose_state()
        for k, v in post.items():
            traj[k].append(v)
        traj['success'].append(_np(task._check_success()).astype(bool))
        traj['keypoint_error'].append(_np((task.keypoints_socket - task.keypoints_plug).norm(dim=-1).mean(dim=-1)))
        traj['contact_8d_right'].append(fields['contact_8d_right'].astype(np.float32))
        traj['state_obs'].append(np.asarray(obs['state'], dtype=np.float32))
        traj['action'].append(action)
        traj['chunk_step'].append(chunk_step)
        traj['n_ext_frames'].append(n_ext)
        traj['diffusion_request'].append(chunk_step == 0)
        traj['latent_changed'].append(changed)
        traj['policy_wall_seconds'].append(dt)
        if lat is not None:
            traj['latent'].append(lat)
        pre = post
        done = np.logical_or(done, np.asarray(reset).astype(bool))
        steps_taken += 1
        if steps_taken % 50 == 0:
            print(f'[usb_eval] step {steps_taken}/{horizon_steps} success_any_so_far '
                  f'{int(np.any(np.stack(traj["success"]), 0).sum())}', flush=True)

    video_paths = env.render() if n_test_vis else []
    client.close()

    T = {k: np.stack(v) for k, v in traj.items()}
    np.savez_compressed(output_path / 'rollout_traj.npz', **T)
    if dres is not None:
        dres.save(output_path / 'delta_log.npz', np.stack(traj['success']))
    np.savez_compressed(output_path / 'refresh_log.npz', **{k: np.asarray(v) for k, v in rlog.items()})

    succ = T['success']
    _, expected = refresh_schedule(steps_taken, latent_refresh)
    lc = T['latent_changed']
    change_steps_all = np.flatnonzero(lc.all(1))
    change_steps_any = np.flatnonzero(lc.any(1))
    refresh_check = dict(
        R=latent_refresh, steps=int(steps_taken), expected_diffusion_calls=int(len(expected)),
        diffusion_requests=int(T['diffusion_request'].sum()),
        latent_change_steps_all_envs=int(len(change_steps_all)),
        latent_change_steps_any_env=int(len(change_steps_any)),
        latent_changes_per_env_min=int(lc.sum(0).min()), latent_changes_per_env_max=int(lc.sum(0).max()),
        change_steps_match_schedule=bool(np.array_equal(change_steps_any, expected)),
        max_n_ext_frames=int(T['n_ext_frames'].max()),
        latent_logged=bool('latent' in T))
    refresh_check['pass'] = bool(refresh_check['diffusion_requests'] == len(expected)
                                 and refresh_check['change_steps_match_schedule'])
    result = dict(success_rate=float(succ.any(0).mean()), terminal_success_rate=float(succ[-1].mean()),
                  flip=flip.tolist(), success_any=succ.any(0).tolist(),
                  steps_taken=int(steps_taken),
                  refresh_check=refresh_check, video_paths=video_paths, robot_command_sent=False)
    (output_path / 'eval_log.json').write_text(json.dumps(result, indent=2) + '\n')
    print('[usb_keyed_bridge] wrote rollout_traj.npz: ' + ', '.join(f'{k}{v.shape}' for k, v in T.items()))
    print(f'[usb_keyed_bridge] result: {result}')


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        for R in (6, 12, 24):
            cs, ex = refresh_schedule(500, R)
            assert len(ex) == math.ceil(500 / R) and cs.max() == R - 1 and np.all(np.diff(ex) == R), R
        print('selftest ok')
        sys.exit(0)
    main()
