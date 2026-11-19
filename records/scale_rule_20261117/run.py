"""scale_rule_20261117 (prereg.json frozen). Offline only."""
import json, hashlib, numpy as np
O = '/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs'; J = f'{O}/scale_rule_20261117'
sha = hashlib.sha256(open(f'{J}/prereg.json', 'rb').read()).hexdigest(); assert sha == open(f'{J}/frozen.sha256').read().split()[0]
E = np.load(f'{O}/link_es_20261112/corr_est.npz'); TH = 1.0
def ridge(obs, P):
    x = np.concatenate([obs.astype(np.float32), P[:, :7].astype(np.float32)], 1); z = np.clip((x[:, E['k']] - E['mu'][E['k']]) / E['sd'][E['k']], -10, 10); return z @ E['wx'] + E['bx'], z @ E['wr'] + E['br']
def counter(ok):
    c = np.zeros(len(ok), int); k = 0
    for i, o in enumerate(ok): k = k + 1 if o else 0; c[i] = k
    return c
def otsu(x, nb=256):
    h, ed = np.histogram(x, nb); c = (ed[:-1] + ed[1:]) / 2; w0 = np.cumsum(h); w1 = w0[-1] - w0; s = np.cumsum(h * c); m0 = s / np.maximum(w0, 1); m1 = (s[-1] - s) / np.maximum(w1, 1)
    v = w0 * w1 * (m0 - m1) ** 2 / w0[-1] ** 2; i = int(np.argmax(v[:-1])); return float(ed[i + 1]), float(v[i] / x.var())
def rule(x):
    x = np.asarray(x, np.float64); tl, el = otsu(np.minimum(x, np.percentile(x, 99))); tg, eg = otsu(np.log(np.maximum(x, 1e-4)))
    pick = 'linear' if el >= eg else 'log'; th = tl if pick == 'linear' else float(np.exp(tg))
    return dict(eta_linear=round(el, 4), eta_log=round(eg, 4), th_linear=round(tl, 4), th_log=round(float(np.exp(tg)), 4), pick=pick, threshold=round(th, 4), share_below=round(float((x < th).mean()), 3))
# theta dev
D = []
for s in (20260921, 20261029, 20261030):
    for c in range(76700, 76900, 25):
        z = np.load(json.load(open(f'{O}/wait_cl_20261110/eval/BAL_s{s}_c{c}/run_metadata.json'))['server_record_path'])
        d = z['dist'][z['has_history'].astype(bool)].astype(np.float64); d = d.transpose(0, 2, 1).reshape(-1, d.shape[1]); D.append(d[d.max(1) > 0].ravel())
theta = rule(np.concatenate(D))
def load(paths):
    ROWS = []
    for r, rec in paths:
        d = np.load(f'{r}/rollout_traj.npz'); su = d['success'].astype(bool); T, N = su.shape; fl = np.load(f'{r}/initial_state.npz')['flip'].astype(bool); ap = d['action_policy']
        z = np.load(rec); si = {int(t): i for i, t in enumerate(z['step'])}; vs = z['verifier_step'].astype(int); P = z['verifier_P']; ch = z['chosen'].astype(int); di = z['dist']
        RX = np.stack([np.stack(ridge(z['obs_features'][si[int(t)]], P[j]), -1) for j, t in enumerate(vs)])
        for e in range(N):
            ok = bool(su[:, e].any()); end = int(np.flatnonzero(su[:, e])[0]) if ok else T; js = [j for j, t in enumerate(vs) if int(t) - 1 < end]
            if not js: continue
            d24 = P[js, e, 10]; cmd24 = np.array([np.linalg.norm(ap[max(0, int(vs[j]) - 24):int(vs[j]), e, :3].sum(0)) * 0.01 for j in js])
            sw = np.array([di[si[int(vs[j])], ch[si[int(vs[j])], e], e] for j in js]) > TH; jsw = int(np.flatnonzero(sw)[0]) if sw.any() else None
            ROWS.append(dict(fl=bool(fl[e]), r=d24 / np.maximum(cmd24, 1e-6), res_ok=(RX[js, e, 0] < 0.5) & (RX[js, e, 1] < 2.9), jsw=jsw))
    return ROWS
def table(ROWS, fn):
    cmax = []
    for x in ROWS:
        c = counter(fn(x)); pre = slice(0, x['jsw']) if x['jsw'] is not None else slice(None); cmax.append(int(c[pre].max()) if len(c[pre]) else 0)
    cmax = np.array(cmax); FL = np.array([x['fl'] for x in ROWS]); tab = {}
    for k in range(1, 5):
        m = cmax >= k
        if m.sum(): tab[k] = dict(n=int(m.sum()), flip_share=round(float(FL[m].mean()), 3))
    ch = {}
    for q in (0.95, 0.90):
        ch[f'q{q}'] = next((k for k in range(1, 6) if (cmax >= k).sum() >= 5 and FL[cmax >= k].mean() >= q), 0)
    return dict(table=tab, chosen=ch)
dev = load([(f'{O}/unify0_20261116/eval/NORULE_s{s}_c{c}', json.load(open(f'{O}/unify0_20261116/eval/NORULE_s{s}_c{c}/run_metadata.json'))['server_record_path']) for s in (20260921, 20260922, 20260923) for c in range(76700, 76900, 25)])
stall_dev = rule(np.concatenate([x['r'] for x in dev])); T_dev = stall_dev['threshold']
H = f'{O}/qdata_20261116/eval'
hold = load([(f'{H}/{a}_s20260921_c{c}', f'{H}/server_records_{a}_s20260921_c{c}.npz') for a in ('Q200_OLD', 'Q400_OLD', 'Q200_NV', 'Q400_NV', 'QFULL_NV') for c in range(76700, 76900, 25)])
stall_hold = rule(np.concatenate([x['r'] for x in hold])); T_h = stall_hold['threshold']
out = dict(study='scale_rule_20261117', prereg_sha256=sha, robot_command_sent=False, theta=theta, stall_dev=stall_dev, stall_holdout=stall_hold, holdout_episodes=len(hold), holdout_flip=int(sum(x['fl'] for x in hold)),
           holdout_dev_threshold=dict(stall_only=table(hold, lambda x: x['r'] < T_dev), evidence=table(hold, lambda x: (x['r'] < T_dev) & x['res_ok'])),
           holdout_own_threshold=dict(stall_only=table(hold, lambda x: x['r'] < T_h), evidence=table(hold, lambda x: (x['r'] < T_h) & x['res_ok'])))
hd = out['holdout_dev_threshold']
out['predictions'] = dict(P1=theta['pick'] == 'linear', P2=stall_dev['pick'] == 'log', P3=max(T_dev, T_h) / max(min(T_dev, T_h), 1e-9) <= 1.5,
                          P4=hd['evidence']['table'].get(2, {}).get('flip_share', 0) >= 0.90, P5=hd['stall_only']['table'].get(2, {}).get('flip_share', 1) < 0.85, P6=hd['evidence']['chosen']['q0.95'] == 2)
json.dump(out, open(f'{J}/summary.json', 'w'), indent=1); print(json.dumps(out, indent=1))
