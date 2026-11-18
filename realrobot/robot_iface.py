"""Robot interfaces for realrobot/runtime.py (2026-11-16).

A robot object provides:
  start_episode()   prepare (zero the tactile baseline, open logging); does NOT move the robot by itself
  read() -> dict    one observation, batch dimension 1:
                      wrist        : (1, 3, 256, 256) float32 in [0, 1], RGB, fixed exposure
                      (front only for the simulation config; the real robot has no external camera, 11-18)
                      state        : (1, 7) float32, hand pose in the robot base frame: x, y, z [m], quaternion
                                     in the SAME order and sign convention as the training data (check before use)
                      contact_8d_right : (1, 8) float32 (xela_c8d.XelaC8D)
                      tactile_raw  : (1, 16, 3) float32 newtons after baseline (logging, residual input)
  send(cmd)         cmd (1, 6) in policy units: [dx, dy, dz] x pos_scale (10 mm per unit per step) and
                    [rx, ry, rz] axis-angle x rot_scale (0.05 rad per unit per step), base frame.
                    Simulation convention: target = MEASURED pose now + cmd (not previous target + cmd).
  success() -> (1,) bool   success sensor (switch at the bottom of the fitting socket)
  end_episode()

FrankaRobot is a stub: the controller is not known yet (user, 2026-11-16: spline output). Points the controller
must satisfy are listed in its docstring. FakeRobot runs the loop without hardware for timing tests.
robot_command_sent: false everywhere in this file.
"""
import time

import numpy as np

POS_SCALE_M = 0.01      # one command unit = 10 mm per control step (simulation pos_action_scale)
ROT_SCALE_RAD = 0.05    # one command unit = 0.05 rad per control step (simulation rot_action_scale)


class FakeRobot:
    """No hardware. Random images, the pose integrates the commands, tactile is zero. For timing / plumbing tests."""

    def __init__(self, seed=0, read_delay_s=0.0, cameras=('wrist',)):
        self.r = np.random.default_rng(seed); self.read_delay_s = read_delay_s; self.sent = []; self.cameras = tuple(cameras)

    def start_episode(self):
        self.pose = np.array([[0.5, 0.0, 0.15, 0.0, 1.0, 0.0, 0.0]], np.float32); self.sent = []

    def read(self):
        if self.read_delay_s:
            time.sleep(self.read_delay_s)
        img = lambda: self.r.random((1, 3, 256, 256), dtype=np.float32)
        out = {c: img() for c in self.cameras}
        out.update(state=self.pose.copy(), contact_8d_right=np.zeros((1, 8), np.float32),
                   tactile_force_field_right=np.zeros((1, 3, 10, 14), np.float32), tactile_raw=np.zeros((1, 16, 3), np.float32))
        return out

    def send(self, cmd):
        cmd = np.asarray(cmd, np.float32); self.sent.append(cmd.copy()); self.pose[:, :3] += cmd[:, :3] * POS_SCALE_M

    def success(self):
        return np.zeros(1, bool)

    def end_episode(self):
        pass


class FrankaRobot:
    """Stub until the controller is specified. Requirements the controller side must meet (from the simulation):

    1. Command reference: target = measured pose + cmd (simulation). If the spline integrates from the previous
       target, cap |target - measured| so that pressing against a surface does not build up force. Use the same
       convention for teleoperated demonstrations and for policy execution.
    2. Smoothing: spline between the 15 Hz commands. Either accept one step of extra latency (simulation: one step
       of observation + one of actuation delay kept success, about 0.7 s slower) or pass the next 2-3 planned
       commands as look-ahead and overwrite them every step.
    3. If no new command arrives in time, decelerate smoothly; never keep the last velocity in contact.
    4. Impedance stiffness and the collision thresholds identical for demonstrations and execution.
    5. Log what the controller actually received (the step command before the spline) - that is the training label.
    """

    def __init__(self, *a, **k):
        raise NotImplementedError('controller interface not specified yet (see docstring)')
