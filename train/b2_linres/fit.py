"""linres_20261117 stage 1 (prereg.json frozen): fit the ridge residual, save linres.npz, offline check on unify0 NORULE rows."""
import sys, json, hashlib, glob, os, numpy as np
sys.path.insert(0, '/home/dyros-recruit/mcy_ws/manifeel/manifeel')
from delta_residual_handframe_20261106 import to_hand
O = '/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs'; J = f'{O}/linres_20261117'
sha = hashlib.sha256(open(f'{J}/prereg.json', 'rb').read()).hexdigest(); assert sha == open(f'{J}/frozen.sha256').read().split()[0]
KEEP_LD = [i for i in range(100) if not (21 <= i < 93 or i == 99)]; RHO = 0.2
files = sorted(glob.glob(f'{O}/usb_keyed_s12r_gatefree_res_20261106/runs/COL_c*/delta_log.npz'))
for S in ('usb_keyed_s13a_vgps_f15_20261107', 'usb_keyed_s13d_mcq_more_20261107', 'usb_keyed_s13e_mcq_t600_20261107'):
    files += [f for f in sorted(glob.glob(f'{O}/{S}/runs/*_c*/delta_log.npz')) if os.path.exists(os.path.join(os.path.dirname(f), 'eval_log.json'))]
N = sum(len(np.load(f)['t']) for f in files); p = 200000 / N; rng = np.random.default_rng(0)
def feats(z, m):
    ldh, yh = to_hand(z['ld'][m].astype(np.float32), z['delta_teacher'][m].astype(np.float32))
    return np.concatenate([ldh[:, KEEP_LD], z['img'][m].astype(np.float32), z['ff'][m].astype(np.float32)], 1), yh
X, Y = [], []
for f in files:
    z = np.load(f); m = rng.random(len(z['t'])) < p
    if m.any(): x, y = feats(z, m); X.append(x); Y.append(y)
X, Y = np.concatenate(X), np.concatenate(Y); print('rows', len(X), flush=True)
mu = X.mean(0); sd = X.std(0) + 1e-6; Xs = (X - mu) / sd; ym = Y.mean(0)
W = np.linalg.solve((Xs.T @ Xs).astype(np.float64) + 10.0 * np.eye(Xs.shape[1]), Xs.T.astype(np.float64) @ (Y - ym)).astype(np.float32); del Xs, X
np.savez(f'{J}/linres.npz', W=W, b=ym.astype(np.float32), mu=mu.astype(np.float32), sd=sd.astype(np.float32), keep_ld=np.array(KEEP_LD), rho=RHO, rows=len(Y))
def predict(z, m):
    x, yh = feats(z, m); return np.clip(((x - mu) / sd) @ W + ym, -RHO, RHO), yh
def head(q):
    x, y, zz, w = (q[..., i] for i in range(4)); h = np.degrees(np.unwrap(np.arctan2(2 * (x * y + w * zz), 1 - 2 * (y * y + zz * zz)), axis=0)); return np.abs((h - h[0:1] + 180) % 360 - 180)
AX = [0, 1, 3, 4, 5]; acc = dict(lin=[], net=[], ylin=[], ynet=[], nolin=[], nonet=[], maglin=[], magnet=[], magt=[])
_, _ = None, None
for s in (20260921, 20260922, 20260923):
    for c in range(76700, 76900, 25):
        r = f'{O}/unify0_20261116/eval/NORULE_s{s}_c{c}'; z = np.load(f'{r}/delta_log.npz'); allm = np.ones(len(z['t']), bool)
        P, YH = predict(z, allm); corr = z['zone'].astype(bool) & (np.abs(z['delta_teacher']).sum(1) > 0)
        _, DH = to_hand(z['ld'].astype(np.float32), z['delta'].astype(np.float32))   # network output in hand frame
        acc['lin'].append(P[corr]); acc['net'].append(DH[corr]); acc['ylin'].append(YH[corr])
        acc['nolin'].append(np.linalg.norm(P[~corr], axis=1)); acc['nonet'].append(np.linalg.norm(DH[~corr], axis=1))
L, Nn, Yc = (np.concatenate(acc[k]) for k in ('lin', 'net', 'ylin'))
def agree(P, Y):
    o = []
    for i in AX:
        nz = (np.abs(Y[:, i]) > 1e-9) & (np.abs(P[:, i]) > 1e-9); o.append(float((np.sign(P[nz, i]) == np.sign(Y[nz, i])).mean()))
    return [round(v, 3) for v in o], round(float(np.mean(o)), 3)
la, lm = agree(L, Yc); na, nm = agree(Nn, Yc)
nolin, nonet = float(np.concatenate(acc['nolin']).mean()), float(np.concatenate(acc['nonet']).mean())
res = dict(stage='1_offline', prereg_sha256=sha, robot_command_sent=False, fit_rows=int(len(Y)), corrected_rows=int(len(Yc)),
           linear=dict(axes_x_y_rx_ry_rz=la, mean=lm, no_correction_mean_norm=round(nolin, 4), corrected_mag_ratio=round(float(np.median(np.linalg.norm(L, axis=1)) / np.median(np.linalg.norm(Yc, axis=1))), 3)),
           network=dict(axes_x_y_rx_ry_rz=na, mean=nm, no_correction_mean_norm=round(nonet, 4), corrected_mag_ratio=round(float(np.median(np.linalg.norm(Nn, axis=1)) / np.median(np.linalg.norm(Yc, axis=1))), 3)))
res['gate'] = 'GO_SIM' if (lm >= 0.68 and nolin <= 1.5 * nonet) else 'NO_SIM'
json.dump(res, open(f'{J}/summary_stage1.json', 'w'), indent=1); print(json.dumps(res, indent=1))
