"""Wrist F/T (wrist_wrench) for the ManiFeel USB task, same estimator as the multi-socket env (new file, 2026-10-27).

The multi-socket env (tacsl_env_insertion_multisocket / tacsl_task_insertion_multisocket, added 2026-10-18) estimates the
external wrench at the flange from the arm joint torques over the control interval, tares it at reset, low-passes it and
adds bias + white noise.  TacSLTaskUSB has none of this.  `install()` registers, in this process only, a subclass of
TacSLTaskUSB whose wrench methods ARE the multi-socket methods (same function objects, not copies), and returns the
`wrist_wrench` config block of TacSLTaskInsertionMultiSocket.yaml so callers can put it into cfg.task.  Library files are
untouched.  Read the measurement from task.wrist_wrench_meas (N, 6) [Fx, Fy, Fz, Tx, Ty, Tz], hand frame.
robot_command_sent: false.
"""
import pathlib

import yaml

HERE = pathlib.Path(__file__).resolve().parent
MS_TASK_YAML = HERE / 'manifeel' / 'config' / 'task' / 'TacSLTaskInsertionMultiSocket.yaml'


def ms_wrist_wrench_cfg(noise_seed=None):
    ww = yaml.safe_load(open(MS_TASK_YAML))['wrist_wrench']
    ww['noise']['seed'] = noise_seed
    return ww


def install(task_map):
    from isaacgymenvs.tasks.tacsl.tacsl_task_USB import TacSLTaskUSB
    from isaacgymenvs.tasks.tacsl.tacsl_env_insertion_multisocket import TacSLEnvInsertionMultiSocket as MSEnv
    from isaacgymenvs.tasks.tacsl.tacsl_task_insertion_multisocket import TacSLTaskInsertionMultiSocket as MSTask

    class TacSLTaskUSBWrench(TacSLTaskUSB):
        _ww_cfg = MSEnv._ww_cfg
        ww_physical_wrench = MSEnv.ww_physical_wrench
        _ww_noise_cfg = MSTask._ww_noise_cfg
        _ww_setup = MSTask._ww_setup
        _ww_randn = MSTask._ww_randn
        _ww_write_obs = MSTask._ww_write_obs
        _ww_reset = MSTask._ww_reset
        _ww_step = MSTask._ww_step

        def __init__(self, cfg, *args, **kwargs):
            self._ww_generator = None
            self.ww_enabled = bool((cfg.get('wrist_wrench') or {}).get('enabled', False))
            self.ww_method = 'joint_torque_external_wrench_estimate'
            super().__init__(cfg, *args, **kwargs)

        def reset_idx(self, env_ids):
            super().reset_idx(env_ids)
            self._ww_reset(env_ids)

        def post_physics_step(self):
            super().post_physics_step()
            self._ww_step()

    task_map['TacSLTaskUSB'] = TacSLTaskUSBWrench
    return TacSLTaskUSBWrench
