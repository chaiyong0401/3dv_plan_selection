"""Episode recorder for the real robot (2026-11-16).

One .npz per episode with every step of the control loop, then build_zarr() writes the replay buffer layout the
training code reads (reactive_diffusion_policy RealImageTactileDataset):
  data/wrist               (T, 256, 256, 3) uint8 RGB   (data/front too if the frame has it; real robot: wrist only)
  data/state               (T, 7) float32
  data/contact_8d_right    (T, 8) float32
  data/action              (T, 6) float32   the step command the controller received (policy units)
  data/source              (T,) int8        20 = human demonstration, 21 = policy, 22 = human take-over (local correction), 23 = human take-over (mode switch)
  data/phase               (T,) int8        0 (kept for the loss-mask convention of the existing datasets)
  meta/episode_ends        (E,) int64
Extra per-step fields (tactile_raw, cmd_policy, cmd_residual, takeover flag, timing) stay in the npz files.
"""
import pathlib
import time

import numpy as np

SOURCE = {'demo': 20, 'policy': 21, 'takeover': 22, 'takeover_switch': 23}   # 22 = local correction (residual labels), 23 = mode switch (generator data only)
CAMERAS = ('wrist', 'front')


class EpisodeRecorder:
    def __init__(self, out_dir):
        self.out = pathlib.Path(out_dir); self.out.mkdir(parents=True, exist_ok=True); self.buf = None

    def start(self, meta):
        self.buf = {'meta': dict(meta, start_time=time.time())}; self.steps = []

    def add(self, frame, label_cmd, kind, **extra):
        """frame: the observation the command was computed from; label_cmd: (6,) command sent to the controller;
        kind: 'demo' | 'policy' | 'takeover' | 'takeover_switch'."""
        img = lambda x: np.clip(np.moveaxis(np.asarray(x[0]), 0, -1) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        s = {c: img(frame[c]) for c in CAMERAS if c in frame}
        s.update(state=np.asarray(frame['state'][0], np.float32),
                 contact_8d_right=np.asarray(frame['contact_8d_right'][0], np.float32), action=np.asarray(label_cmd, np.float32).reshape(6),
                 source=np.int8(SOURCE[kind]), t_wall=np.float64(time.time()))
        for k, v in extra.items():
            s[k] = np.asarray(v)
        self.steps.append(s)

    def finish(self, success, note=''):
        keys = self.steps[0].keys()
        arr = {k: np.stack([s[k] for s in self.steps]) for k in keys}
        name = time.strftime('%Y%m%d_%H%M%S')
        np.savez_compressed(self.out / f'ep_{name}.npz', **arr, success=np.bool_(success), note=np.str_(note), meta=np.str_(repr(self.buf['meta'])))
        return self.out / f'ep_{name}.npz'


def build_zarr(npz_paths, zarr_path):
    import zarr
    eps = [np.load(p) for p in npz_paths]
    root = zarr.open(str(pathlib.Path(zarr_path) / 'replay_buffer.zarr'), 'w')
    data = root.create_group('data'); meta = root.create_group('meta')
    for k in [c for c in CAMERAS if c in eps[0].files] + ['state', 'contact_8d_right', 'action', 'source']:
        a = np.concatenate([e[k] for e in eps], 0)
        data.create_dataset(k, data=a, chunks=(64,) + a.shape[1:] if a.ndim > 1 else (4096,))
    n = sum(len(e['action']) for e in eps)
    data.create_dataset('phase', data=np.zeros(n, np.int8))
    meta.create_dataset('episode_ends', data=np.cumsum([len(e['action']) for e in eps]).astype(np.int64))
    return root
