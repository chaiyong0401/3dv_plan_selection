"""rr_poolB_20261123: lifting candidates of one generator at wrong-socket (unseen) and right-socket stall scenes.
usage: poolB_20261123.py <ldp ckpt> <tag> <out json>. Offline, existing records only, no robot. Definitions = 5090
diag_leave_20261123/manyk.py (8 seeds x 16 candidates, lift = mean z command over the first 12 steps > 0.06)."""
import json, os, sys
import numpy as np, torch
W = '/home/dyros-recruit/mcy_ws/real_ms_4070'; C = W + '/code_5090_20261123'; D = W + '/data/poolB_20261123'
sys.path[:0] = [C + '/realrobot', C + '/rdp_side']
from policy_core import PolicyCore
from sensors_zmq import decode_resize
core = PolicyCore(sys.argv[1], W + '/at_20261120/checkpoints/latest.ckpt', None, q='none', record=True, evid=None, task='real_ms_c8d_ldp_wrist', select_k=16)
un = core.policy.normalizer['action'].unnormalize; H = core.obs_steps * core.ratio + 2
def plan(h, seed):
    core.reset(1)
    for k in range(len(h['im'])):
        core.push_frame({'wrist': h['im'][k][None], 'state': h['state'][k][None], 'contact_8d_right': h['c'][k][None]}, fill=(k == 0))
    _, info = core.plan(core.snapshot(), seed=seed)
    return un(torch.as_tensor(info['plan_full'].astype(np.float32)).reshape(16, 1, -1, 6).to(core.device)).cpu().numpy()[:, 0]
def hist(e, t):
    ends = e['wrist_jpeg_ends']; st = np.r_[0, ends[:-1]]; a = max(0, t - H)
    return dict(im=[np.moveaxis(decode_resize(e['wrist_jpeg'][st[k]:ends[k]], (240, 426)).astype(np.float32) / 255, -1, 0) for k in range(a, t + 1)], state=e['state'][a:t + 1].copy(), c=e['contact_8d_right'][a:t + 1])
def lifts(e, t):
    h = hist(e, t); return sum(int((plan(h, 100 + s)[:, :12, 2].mean(1) > 0.06).sum()) for s in range(8))
rows = []
for r in sorted(json.load(open(D + '/split.json'))['episodes'], key=lambda r: r['name']):
    if not r['valid']: continue
    e = np.load(D + '/merged/' + r['name'], allow_pickle=True)
    grp = 'R' if r['kind'] == 'ALIGN' else ('W_unseen' if r['held_out'] else 'W_train')
    n = lifts(e, r['handoff_step'] - 1); rows.append(dict(scene=r['name'][12:18], group=grp, correct=r['correct'], lift=n, share=n / 128))
    print(sys.argv[2], rows[-1], flush=True)
for name in ('ep_20261123_210725_0001.npz', 'ep_20261123_210845_0003.npz'):
    e = np.load(D + '/jit_res/episodes/' + name, allow_pickle=True); T = len(e['state']); ns = [lifts(e, t) for t in (400, 650, 890) if t < T]
    rows.append(dict(scene=name[12:18] + '_night', group='W_unseen', correct='-y', lift=ns, share=float(np.mean(ns)) / 128, steps=T)); print(sys.argv[2], rows[-1], flush=True)
json.dump(dict(tag=sys.argv[2], ckpt=sys.argv[1], legacy_clip=os.environ.get('RDP_PHOTO_JITTER_LEGACY_CLIP') == '1', rows=rows, robot_command_sent=False), open(sys.argv[3], 'w'), indent=1)
