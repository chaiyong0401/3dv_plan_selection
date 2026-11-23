"""rr_poolB_20261123 verdict from the two per-model JSONs. usage: analyze_poolB_20261123.py <jit json> <tkL json> <out json>"""
import json, sys
import numpy as np
def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg); return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))
out = {}
for tag, f in (('jit', sys.argv[1]), ('tkL', sys.argv[2])):
    rows = json.load(open(f))['rows']; g = {k: [r for r in rows if r['group'] == k] for k in ('W_unseen', 'R', 'W_train')}
    s = {k: [r['share'] for r in v] for k, v in g.items()}
    out[tag] = dict(n={k: len(v) for k, v in g.items()}, supply_scene_share={k: round(float(np.mean(np.array(v) >= 1 / 16)), 3) for k, v in s.items()},
                    mean_share={k: round(float(np.mean(v)), 3) for k, v in s.items()}, auroc_Wunseen_vs_R=round(auroc(s['W_unseen'], s['R']), 3),
                    W_unseen=[(r['scene'], r['correct'], r['lift']) for r in g['W_unseen']], R=[(r['scene'], r['correct'], r['lift']) for r in g['R']])
t = out['tkL']; P1 = t['supply_scene_share']['W_unseen'] >= 0.5; P2 = t['supply_scene_share']['R'] <= 0.2; P3 = t['auroc_Wunseen_vs_R'] >= 0.75
out['criteria'] = dict(P1_supply=bool(P1), P2_quiet=bool(P2), P3_separation=bool(P3))
out['verdict'] = 'B_OK_WITHOUT_SELECTOR' if (P1 and P2 and P3) else ('B_NEEDS_SELECTOR' if P1 else 'B_SUPPLY_SHORT')
out['robot_command_sent'] = False
json.dump(out, open(sys.argv[3], 'w'), indent=1); print(json.dumps({k: v for k, v in out.items() if k in ('criteria', 'verdict')}))
