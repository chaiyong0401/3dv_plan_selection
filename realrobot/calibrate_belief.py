"""Calibration of the belief rule from policy rollouts without task labels (2026-11-16).

choose_k: promotion count = the smallest evidence count k whose episodes are wrong-mode in at least `q` of
cases (q = 0.95: at most 5 % of forced switches are unnecessary), with at least `n_min` episodes reaching k.
The wrong-mode label is outcome-based: a scene is right-mode if some rollout of it succeeded without switching
(repeat each scene a few times). On the USB simulation logs k = 2 for any q in 0.85 .. 0.99.
"""
import numpy as np


def choose_k(c_max, wrong, q=0.95, n_min=5, k_max=5):
    """c_max: (E,) max consecutive evidence count before the first switch per episode; wrong: (E,) bool."""
    c_max, wrong = np.asarray(c_max), np.asarray(wrong, bool)
    for k in range(1, k_max + 1):
        m = c_max >= k
        if m.sum() >= n_min and wrong[m].mean() >= q:
            return k
    return 0


def outcome_wrong(scene_ids, success, switched):
    """Per-episode wrong-mode label from repeated rollouts of the same scenes."""
    scene_ids, success, switched = map(np.asarray, (scene_ids, success, switched))
    right = {s: bool(np.any(success[scene_ids == s] & ~switched[scene_ids == s])) for s in np.unique(scene_ids)}
    return np.array([not right[s] for s in scene_ids])


if __name__ == '__main__':
    assert choose_k([0, 1, 2, 2, 2, 2, 2, 3], [0, 0, 1, 1, 1, 1, 1, 1]) == 2
    assert outcome_wrong([1, 1, 2, 2], [1, 0, 1, 1], [0, 1, 1, 1]).tolist() == [False, False, True, True]
    print('calibrate_belief selftest ok')
