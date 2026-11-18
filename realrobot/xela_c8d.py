"""XELA uSPA44 (4 x 4 taxels x 3 axes) -> the policy's 8-value tactile summary
C8D = [Fx, Fy, Fz, Tx, Ty, Tz, MaxForce, ContactArea], computed the same way as the simulation converter
(scripts/convert_force_field_to_c8d.py), which was itself ported from the original uSPA44 pipeline
(SPP_RAD_IROS2026/create_hybrid_dataset_v2.py: XELA_USPA44_COORD, calculate_soft_area).

Steps on the robot:
  1. read the 16 x 3 raw values (sensor units) and convert to newtons with the sensor's calibration,
  2. subtract the baseline taken after grasping, before contact (zero_from()),
  3. compute C8D.
Axis order must be checked on the real sensor: index 2 is the normal (pressing) axis, 0 / 1 the shear axes in the
pad plane, consistent with the coordinate table below. The soft-area threshold tau is a parameter (the
simulation used 0.005 for its unit scale; the original uSPA44 pipeline used 0.1 ~ 0.5 N depending on the task)
and has to be set from real contact data. The policy's normalizer was fitted on simulated C8D, so a new
normalizer (and new training) on real data is required anyway; this module only fixes the definition.
"""
import numpy as np

# taxel centres in metres, pad frame (x, y, 0), 10 mm pitch, row-major, from SPP_RAD_IROS2026/create_hybrid_dataset_v2.py
XELA_USPA44_COORD = np.array([[x, y, 0.0] for x in (-1.5, -0.5, 0.5, 1.5) for y in (-1.5, -0.5, 0.5, 1.5)]) / 100.0


def soft_area(fz, tau, alpha):
    return np.sum(1.0 / (1.0 + np.exp(-alpha * (fz - tau))), axis=-1, keepdims=True)


class XelaC8D:
    def __init__(self, tau=0.25, alpha=80.0, coords=XELA_USPA44_COORD):
        self.tau, self.alpha, self.coords = float(tau), float(alpha), np.asarray(coords, np.float64)
        self.base = None

    def zero_from(self, frames):
        """frames: (T, 16, 3) newtons recorded after grasping and before contact; their mean becomes the baseline."""
        self.base = np.asarray(frames, np.float64).reshape(-1, 16, 3).mean(0)

    def __call__(self, f16x3):
        """f16x3: (..., 16, 3) newtons [shear x, shear y, normal]. Returns (..., 8) float32."""
        f = np.asarray(f16x3, np.float64)
        if self.base is not None:
            f = f - self.base
        F = f.sum(-2)
        T = np.cross(np.broadcast_to(self.coords, f.shape), f).sum(-2)
        mx = f[..., 2].max(-1, keepdims=True)
        area = soft_area(f[..., 2], self.tau, self.alpha)
        return np.concatenate([F, T, mx, area], -1).astype(np.float32)


if __name__ == '__main__':
    c = XelaC8D()
    z = np.zeros((16, 3)); assert np.allclose(c(z)[:6], 0)
    p = z.copy(); p[0, 2] = 1.0          # 1 N on the corner taxel (-15 mm, -15 mm)
    out = c(p); assert np.isclose(out[2], 1.0) and np.isclose(out[3], -0.015) and np.isclose(out[4], 0.015), out
    print('xela_c8d selftest ok', out.round(4).tolist())
