"""usb_keyed_s13a_vgps_f15_20261107 = s8 build_dataset.py with runs = this experiment (F15 + residual_HF deploy_gf, K16, select-horizon 24 so each
candidate = plan_full 24x6 + dist + 1 = 146), --runs_glob for the smoke dry-run. Original s8 docstring: refresh-level offline RL dataset for the USB value selector (training order step 5).
= multisocket_vgps_20261031/build_dataset.py adapted to USB: runs = usb_keyed_s7_seldata_20261102 runs (F5 + s6 residual deploy,
K16, horizon 8, rule first / random alternating per cohort); same row / transition definition:
rows = (env, refresh) with step t >= 36 and t before the env's first success; P_generic (28) rebuilt with the server formula
(state at the request = state_obs[t-1], d24, d12 (+norms), mean / std of action_policy t-12..t-1, t/499); 8 candidates = executed +
7 others at random of K16, each plan_full (8x6 normalized, flattened) + dist + 1; r = 1 and done if the first success falls in
[t, t_next); done with r = 0 if there is no next refresh. Extra per-row fields for the offline gate: behaviour rule (0 first,
1 random), executed index in K16, flip, and all 16 candidates (C16) for the K16 argmax used by the IPS check.
Usage: build_dataset.py [--max_runs n] [--out dir]"""
import argparse, json, glob, os, sys, collections, numpy as np
sys.path.insert(0, '/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs/multisocket_vgps_20261031')
from build_dataset import pg_rows          # verbatim server formula (multisocket VGPS, verified there to 6e-8)

HERE = os.path.dirname(os.path.abspath(__file__))
S7 = HERE
KC = 8
ap = argparse.ArgumentParser(); ap.add_argument('--max_runs', type=int, default=0); ap.add_argument('--out', default=f'{HERE}/data'); ap.add_argument('--runs_glob', default='first_c*,random_c*')
args = ap.parse_args()


def main():
    rng = np.random.default_rng(20261102)
    runs = []
    for rd in sorted(sum([glob.glob(f'{S7}/runs/{g}') for g in args.runs_glob.split(',')], [])):
        tag = os.path.basename(rd); rec = f'{S7}/runs/server_records_{tag}.npz'
        if os.path.exists(f'{rd}/eval_log.json') and os.path.exists(f'{rd}/rollout_traj.npz') and os.path.exists(rec):
            runs.append((tag, rd, rec))
    runs.sort(key=lambda x: x[0].rsplit('_c', 1)[-1])
    if args.max_runs: runs = runs[:args.max_runs]
    F = collections.defaultdict(list); meta = []; rej = collections.Counter()
    for ri, (tag, rd, rec) in enumerate(runs):
        m = json.load(open(f'{rd}/run_metadata.json'))
        assert m['delta_mode'] == 'deploy_gf' and m['socket_flip'] and m['max_steps'] == 400, tag
        beh = 1 if 'random' in tag else 0
        z = np.load(rec); d = np.load(f'{rd}/rollout_traj.npz')
        flip = np.load(f'{rd}/initial_state.npz')['flip'].astype(bool)
        state, act, succ = d['state_obs'], d['action_policy'], d['success'].astype(bool)
        steps, chosen, dist, pf, of = z['step'], z['chosen'], z['dist'], z['cand_plan_full'], z['obs_features']
        R, K, N = dist.shape; T = succ.shape[0]
        assert K == 16 and state.shape[1] == N, tag
        first = np.array([np.flatnonzero(succ[:, e])[0] if succ[:, e].any() else 10 ** 6 for e in range(N)])
        idx, rows = {}, []
        for r, t in enumerate(steps):
            t = int(t)
            if t < 36 or t >= T: continue
            envs = np.flatnonzero(first > t)
            if len(envs) == 0: continue
            if not np.isfinite(dist[r][:, envs]).all(): rej['nan_dist_rows'] += len(envs); continue
            P = pg_rows(state, act, t, envs)
            for j, e in enumerate(envs):
                ch = int(chosen[r, e])
                sel = np.r_[ch, rng.choice(np.setdiff1d(np.arange(K), [ch]), KC - 1, replace=False)]
                full = np.concatenate([pf[r, :, e].reshape(K, -1).astype(np.float32), dist[r, :, e, None],
                                       np.ones((K, 1), np.float32)], 1)
                idx[(r, e)] = len(rows)
                rows.append((of[r, e], P[j], full[sel].astype(np.float16), full.astype(np.float16), t, e, r, ch))
        if not rows: continue
        g0 = len(F['t'])
        for (o, P, cand, c16, t, e, r, ch) in rows:
            F['V'].append(o); F['P'].append(P); F['C'].append(cand); F['C16'].append(c16); F['t'].append(t)
            F['env'].append(e); F['run'].append(ri); F['beh'].append(beh); F['exec16'].append(ch); F['flip'].append(int(flip[e]))
            nxt, rew, done = -1, 0.0, 1.0
            rr = r + 1
            tn = int(steps[rr]) if rr < R else T
            if first[e] < tn:
                rew = 1.0
            elif rr < R and (rr, e) in idx:
                nxt, done = g0 + idx[(rr, e)], 0.0
            F['next'].append(nxt); F['rew'].append(rew); F['done'].append(done)
            F['win48'].append(float(first[e] < t + 48)); F['ep_succ'].append(float(first[e] < 10 ** 6))
        meta.append({'run_index': ri, 'tag': tag, 'run': rd, 'record': rec, 'behaviour': 'random' if beh else 'first', 'rows': len(rows)})
    os.makedirs(args.out, exist_ok=True)
    dt = {'V': np.float16, 'P': np.float32, 'C': np.float16, 'C16': np.float16, 't': np.int32, 'env': np.int32, 'run': np.int32,
          'beh': np.int8, 'exec16': np.int8, 'flip': np.int8, 'next': np.int64, 'rew': np.float32, 'done': np.float32,
          'win48': np.float32, 'ep_succ': np.float32}
    for k, v in F.items():
        np.save(f'{args.out}/{k}.npy', np.asarray(v, dtype=dt[k]))
    summ = {'n_rows': len(F['t']), 'n_runs_used': len(meta), 'by_behaviour': collections.Counter(m['behaviour'] for m in meta),
            'rejected': dict(rej), 'reward_rows': float(np.sum(F['rew'])), 'done_rows': float(np.sum(F['done']))}
    json.dump({'summary': summ, 'runs': meta}, open(f'{args.out}/manifest.json', 'w'), indent=1, default=int)
    print(json.dumps(summ, default=int))


if __name__ == '__main__':
    main()
