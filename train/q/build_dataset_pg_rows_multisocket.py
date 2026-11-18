"""multisocket_vgps_20261031: refresh-level offline RL dataset for the V-GPS-style baseline.

Rows = (env, refresh) with step t >= 36 (the server scores candidates only from step 36 on) and t before the env's
first success. Per row: obs_features (2078, fp16, as recorded), P_generic (28, reconstructed from rollout_traj with
the server formula: state at the request [:7], d24 = its change over 24 steps (+norm), d12 (+norm), mean / std of the policy actions
t-12..t-1, t/499), 8 candidates (executed one first + 7 others at random for K16; all 8 for K8), each
plan_full (12x6 normalized, flattened) + dist + 1 (= the server's rawplan candidate vector), reward / done / next row.
Transition: r = 1 and done if the first success falls in [t, t_next); done with r = 0 if there is no next refresh.

Runs: survey.py filter (MS6 base, no eval / smoke cohort, not stage1 / stage2 / ft / ablation studies), widened to
deploy-time residual runs (the data the design draft names, e.g. cand_q_explore = rotres_deploy); runs where a teacher
or oracle acts (delta_mode *collect*, oracle_mode != off, takeover bridges) are excluded.
"""
import json, glob, os, sys, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from survey import D, EXCL_STUDY, cohort_ok

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
KC = 8
GAMMA_NOTE = 'reward 1 at success, gamma applied in training'


def run_ok(m):
    why = []
    if 'ms6tko_ftF43_1020' not in m.get('policy', {}).get('ldp_checkpoint', ''): why.append('base')
    dm = str(m.get('delta_mode', 'off'))
    if 'collect' in dm: why.append('collect')
    if m.get('bres_mode', 'off') not in ('off', 'none', None): why.append('bres')
    if m.get('oracle_mode', 'off') not in ('off', None): why.append('oracle')
    if 'takeover' in m.get('bridge', ''): why.append('takeover')
    return why


def pg_rows(state, act, t, envs):
    # rollout_traj state_obs[i] is appended after step i's action; the server's request at step t holds state_obs[t-1]
    # (stt), pose_history[-24] = state_obs[t-25], pose_history[-12] = state_obs[t-13] (checked on the dev smoke records)
    s = state[t - 1, envs]
    d24 = s[:, :3] - state[t - 25, envs, :3]
    d12 = s[:, :3] - state[t - 13, envs, :3]
    a = act[t - 12:t][:, envs]                     # (12, n, 6)
    return np.concatenate([s[:, :7], d24, np.linalg.norm(d24, axis=-1, keepdims=True), d12,
                           np.linalg.norm(d12, axis=-1, keepdims=True), a.mean(0), a.std(0),
                           np.full((len(envs), 1), t / 499.0)], 1).astype(np.float32)


def main():
    rng = np.random.default_rng(20261031)
    runs, rej = [], collections.Counter()
    for md in sorted(glob.glob(f'{D}/multisocket_*/**/run_metadata.json', recursive=True)):
        st = md[len(D) + 1:].split('/')[0]
        if st in EXCL_STUDY or '/dev_runs/' in md or 'smoke' in md: rej['study/dev'] += 1; continue
        m = json.load(open(md)); c = int(m.get('test_start_seed', -1))
        if not cohort_ok(c): rej['cohort'] += 1; continue
        why = run_ok(m)
        if why: rej['+'.join(why)] += 1; continue
        rec, rd = m.get('server_record_path'), os.path.dirname(md)
        if not rec or not os.path.exists(rec) or not os.path.exists(f'{rd}/rollout_traj.npz'): rej['missing'] += 1; continue
        runs.append((st, rd, rec, c, str(m.get('delta_mode'))))
    print('candidate runs', len(runs), 'rejected', dict(rej), flush=True)
    F = collections.defaultdict(list); meta = []
    for ri, (st, rd, rec, c, dm) in enumerate(runs):
        try:
            z = np.load(rec)
            if 'cand_plan_full' not in z.files or 'obs_features' not in z.files: rej['no_plan_full'] += 1; continue
            d = np.load(f'{rd}/rollout_traj.npz')
            state, act, succ = d['state_obs'], d['action_policy'], d['success'].astype(bool)
            steps, chosen, dist = z['step'], z['chosen'], z['dist']
            pf, of = z['cand_plan_full'], z['obs_features']
        except Exception as e:
            rej['bad_npz'] += 1; print('bad', rec, e, flush=True); continue
        R, K, N = dist.shape
        if K not in (8, 16) or state.shape[1] != N: rej[f'shape_K{K}'] += 1; continue
        T = succ.shape[0]
        first = np.array([np.flatnonzero(succ[:, e])[0] if succ[:, e].any() else 10 ** 6 for e in range(N)])
        idx = {}
        rows = []
        for r, t in enumerate(steps):
            t = int(t)
            if t < 36 or t >= T: continue
            envs = np.flatnonzero(first > t)
            if len(envs) == 0: continue
            if not np.isfinite(dist[r][:, envs]).all(): rej['nan_dist_rows'] += len(envs); continue
            P = pg_rows(state, act, t, envs)
            for j, e in enumerate(envs):
                ch = int(chosen[r, e])
                sel = np.r_[ch, np.setdiff1d(np.arange(K), [ch])] if K == 8 else \
                    np.r_[ch, rng.choice(np.setdiff1d(np.arange(K), [ch]), KC - 1, replace=False)]
                cand = np.concatenate([pf[r, sel, e].reshape(KC, -1).astype(np.float32), dist[r, sel, e, None],
                                       np.ones((KC, 1), np.float32)], 1).astype(np.float16)
                idx[(r, e)] = len(rows)
                rows.append((of[r, e], P[j], cand, t, e, r))
        if not rows: continue
        g0 = len(F['t'])
        for (o, P, cand, t, e, r) in rows:
            F['V'].append(o); F['P'].append(P); F['C'].append(cand); F['t'].append(t)
            F['env'].append(e); F['run'].append(ri)
            # next refresh with a row for this env
            nxt, rew, done = -1, 0.0, 1.0
            rr = r + 1
            tn = int(steps[rr]) if rr < R else T
            if first[e] < tn:
                rew = 1.0
            elif rr < R and (rr, e) in idx:
                nxt, done = g0 + idx[(rr, e)], 0.0
            F['next'].append(nxt); F['rew'].append(rew); F['done'].append(done)
            F['win48'].append(float(first[e] < t + 48))
        meta.append({'run_index': ri, 'study': st, 'run': rd, 'record': rec, 'cohort': c, 'delta_mode': dm, 'K': int(K),
                     'rows': len(rows)})
        if ri % 50 == 0: print(ri, len(runs), 'rows', len(F['t']), flush=True)
    os.makedirs(OUT, exist_ok=True)
    dt = {'V': np.float16, 'P': np.float32, 'C': np.float16, 't': np.int32, 'env': np.int32, 'run': np.int32,
          'next': np.int64, 'rew': np.float32, 'done': np.float32, 'win48': np.float32}
    for k, v in F.items():
        np.save(f'{OUT}/{k}.npy', np.asarray(v, dtype=dt[k]))
    summ = {'n_rows': len(F['t']), 'n_runs_used': len(meta), 'rejected': dict(rej),
            'by_study': collections.Counter(m['study'] for m in meta), 'by_delta_mode': collections.Counter(m['delta_mode'] for m in meta),
            'by_K': collections.Counter(m['K'] for m in meta), 'reward_rows': float(np.sum(F['rew'])),
            'done_rows': float(np.sum(F['done']))}
    json.dump({'summary': summ, 'runs': meta}, open(f'{OUT}/manifest.json', 'w'), indent=1, default=int)
    print(json.dumps(summ, indent=1, default=int))


if __name__ == '__main__':
    main()
