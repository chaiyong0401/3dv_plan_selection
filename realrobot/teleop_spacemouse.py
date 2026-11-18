"""3D mouse -> step command in policy units (2026-11-16).

The policy's command is a per-step displacement: [dx, dy, dz] in units of 10 mm and [rx, ry, rz] in units of
0.05 rad per control step (15 Hz), clipped to [-1, 1], base frame. The 3D mouse gives a 6-axis deflection;
we map full deflection to `max_unit` (default 0.6 of the policy limit, so demonstrations stay inside the range the
policy can reproduce), with a dead band against drift and an optional per-axis sign / permutation to align the
mouse frame with the robot base frame (set once with the robot in front of you).

Buttons: left = toggle take-over (policy <-> human), right = end of episode.
The value returned by command() is exactly what goes to the controller and is what the recorder stores.

Backend: pyspacemouse (pip install pyspacemouse; needs hidapi and udev access). Not imported until used.
"""
import numpy as np


class SpaceMouseTeleop:
    def __init__(self, max_unit=0.6, dead=0.08, axis_map=(0, 1, 2, 3, 4, 5), axis_sign=(1, 1, 1, 1, 1, 1), expo=1.5):
        self.max_unit, self.dead, self.expo = float(max_unit), float(dead), float(expo)
        self.axis_map, self.axis_sign = list(axis_map), np.asarray(axis_sign, np.float32)
        self.takeover = False; self.end_requested = False; self._prev = (0, 0); self.dev = None

    def open(self):
        import pyspacemouse
        ok = pyspacemouse.open()
        if not ok:
            raise RuntimeError('3D mouse not found (pyspacemouse.open() failed)')
        self.dev = pyspacemouse

    def _shape(self, v):
        v = np.where(np.abs(v) < self.dead, 0.0, (np.abs(v) - self.dead) / (1 - self.dead) * np.sign(v))
        return np.sign(v) * np.abs(v) ** self.expo          # finer control near zero

    def command(self):
        s = self.dev.read()
        raw = np.array([s.x, s.y, s.z, s.roll, s.pitch, s.yaw], np.float32)
        b = (int(s.buttons[0]) if len(s.buttons) > 0 else 0, int(s.buttons[1]) if len(s.buttons) > 1 else 0)
        if b[0] and not self._prev[0]:
            self.takeover = not self.takeover
        if b[1] and not self._prev[1]:
            self.end_requested = True
        self._prev = b
        v = self._shape(raw[self.axis_map] * self.axis_sign)
        return np.clip(v * self.max_unit, -1.0, 1.0).astype(np.float32).reshape(1, 6)
