"""nowr_cl_20261112 train = link_es train_conv2.py with ONE change: input columns 21:93 (wrist wrench history) and 99 (steps since the wrench-detected contact event) are blinded (mu 0, sd 1e9). link_es docstring: link_es_20261112 train = conv_cls train_conv.py with ONE change: the validation loss used for early stopping / best epoch is the residual loss only (the BCE head term is excluded), as in HF2. Original docstring: conv_cls_20261111 train (head = two logits "correction needed" xy / rot, BCE weight 0.02; replaces the magnitude regression of conv_head): = usb_keyed_s12z8 train_dagger.py (same data, split, HP, residual loss) + a convergence head on the shared trunk:
q = RHO*sigmoid(Linear(256,2)) regressing the remaining-correction magnitude (|teacher delta xy|, |teacher delta rot|, hand frame; out-of-zone rows = RHO), loss added in the same units and weights. See prereg.json."""
import glob, hashlib, json, os, pathlib, sys
import numpy as np, torch
J = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(J.parents[2])); SRCJ = J.parent / 'usb_keyed_s12z8_rotflip_res_dagger_20261107'
from delta_residual_ms_rld_20261024 import ResMLP, LD, IMG, FF, RHO
from delta_residual_handframe_20261106 import to_hand
sha = hashlib.sha256((J / 'prereg.json').read_bytes()).hexdigest()
class ResConv(ResMLP):
    def __init__(self, d):
        super().__init__(d=d); self.head = torch.nn.Linear(256, 2)
    def both(self, x):
        h = self.net[:-1]((x - self.mu) / self.sd); return RHO * torch.tanh(self.net[-1](h)), self.head(h)
HP = dict(lr=3e-4, wd=1e-4, batch=256, max_epochs=60, patience=8, dense_after=12, dense_w=4.0, val_frac=0.2, seed=0)
MAXF = int(os.environ.get('MAXF', '0'))   # presmoke only: limit files per source
r4 = lambda x: None if x is None else round(float(x), 4)
O = J.parent
files = sorted(glob.glob(str(O / 'usb_keyed_s12r_gatefree_res_20261106' / 'runs/COL_c*/delta_log.npz')))[:MAXF or None]
src = ['COL'] * len(files)
for S in ('usb_keyed_s13a_vgps_f15_20261107', 'usb_keyed_s13d_mcq_more_20261107', 'usb_keyed_s13e_mcq_t600_20261107'):
    fs = [f for f in sorted(glob.glob(str(O / S / 'runs/*_c*/delta_log.npz'))) if os.path.exists(os.path.join(os.path.dirname(f), 'eval_log.json'))][:MAXF or None]
    files += fs; src += [S.split('_')[2]] * len(fs)
n = [len(np.load(f)['t']) for f in files]; N = int(sum(n)); DX = LD + IMG + FF
print('files', len(files), 'rows', N, flush=True)
X = np.empty((N, DX), np.float16); Y = np.empty((N, 6), np.float32); W = np.empty(N, np.float32); G = np.empty(N, np.int64)
Z = np.empty(N, bool); ROT = np.empty(N, np.float32); SRC = np.empty(N, '<U4'); o = 0
for gi, f in enumerate(files):
    z = np.load(f); t, e = z['t'], z['e']; dense = np.zeros(len(t), bool)
    for ee in np.unique(e):
        m = np.flatnonzero(e == ee); starts = t[m][z['corr_start'][m]]
        for t0 in starts:
            dense[m[(t[m] >= t0) & (t[m] < t0 + HP['dense_after'])]] = True
    ldh, yh = to_hand(z['ld'].astype(np.float32), z['delta_teacher'].astype(np.float32)); k = len(t); s = slice(o, o + k)
    X[s, :LD] = ldh; X[s, LD:LD + IMG] = z['img']; X[s, LD + IMG:] = z['ff']; Y[s] = yh
    W[s] = z['sample_w'] * np.where(dense, HP['dense_w'], 1.0); G[s] = gi * 1000 + e; Z[s] = z['zone']; ROT[s] = z['rot_err_deg']; SRC[s] = src[gi]; o += k
assert np.isfinite(X).all()
MAG = np.stack([np.minimum(np.linalg.norm(Y[:, :2], axis=1), RHO), np.minimum(np.linalg.norm(Y[:, 3:6], axis=1), RHO)], 1).astype(np.float32); MAG[~Z] = RHO
Cxy = np.abs(Y[:, :2]).sum(1) > 0; Crot = np.abs(Y[:, 3:6]).sum(1) > 0; C = Cxy | Crot
NEED = np.stack([~Z | Cxy, ~Z | Crot], 1).astype(np.float32); LAM = 0.02; bce = torch.nn.functional.binary_cross_entropy_with_logits
print('rows', N, 'zone', int(Z.sum()), 'corr_xy', int(Cxy.sum()), 'corr_rot', int(Crot.sum()), 'groups', len(set(G)), flush=True)
rng = np.random.default_rng(HP['seed']); ug = np.unique(G); rng.shuffle(ug); vg = set(ug[:int(len(ug) * HP['val_frac'])])
va = np.isin(G, list(vg)); tr = ~va; tri, vai = np.flatnonzero(tr), np.flatnonzero(va); dev = 'cuda:0'


def cosm(P, Yv, sel, sl):
    if not sel.any(): return None
    a, b = P[sel][:, sl], Yv[sel][:, sl]
    return r4(((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)).mean())


torch.manual_seed(HP['seed'])
mu = np.zeros(DX); sq = np.zeros(DX)
for b in range(0, len(tri), 65536):
    xb = X[tri[b:b + 65536]].astype(np.float64); mu += xb.sum(0); sq += (xb ** 2).sum(0)
mu /= len(tri); sd = np.sqrt(np.maximum(sq / len(tri) - mu ** 2, 0))
BLIND = list(range(21, 93)) + [99]; mu[BLIND] = 0.0; sd[BLIND] = 1e9   # nowr_cl: wrist-wrench history and the wrench-derived contact-event slot are removed from the input
m = ResConv(d=DX).to(dev); m.mu.copy_(torch.tensor(mu, dtype=torch.float32)); m.sd.copy_(torch.tensor(sd + 1e-3, dtype=torch.float32))
Xt = torch.from_numpy(X).to(dev); Yt, Wt = torch.tensor(Y, device=dev), torch.tensor(W, device=dev); Mt = torch.tensor(NEED, device=dev); del X
opt = torch.optim.Adam(m.parameters(), HP['lr'], weight_decay=HP['wd'])


def loss_of(idx, train_):
    m.train(train_); tot, ws = 0.0, 0.0
    for b in range(0, len(idx), HP['batch']):
        k = torch.as_tensor(idx[b:b + HP['batch']], device=dev)
        with torch.set_grad_enabled(train_):
            p, q = m.both(Xt[k].float()); w = Wt[k]; lres = (w[:, None] * (p - Yt[k]) ** 2).sum(-1).sum() / w.sum(); l = lres + LAM * (w[:, None] * bce(q, Mt[k], reduction='none')).sum(-1).sum() / w.sum()
            if train_: opt.zero_grad(); l.backward(); opt.step()
        tot += float(l if train_ else lres) * float(w.sum()); ws += float(w.sum())   # link_es: validation / early stopping on the residual loss only (HF2's criterion)
    return tot / ws


best, bad, hist, sdict = 1e9, 0, [], None
for ep in range(HP['max_epochs'] if not MAXF else 1):
    lt = loss_of(np.random.default_rng(ep).permutation(tri), True); lv = loss_of(vai, False); hist.append([ep, r4(lt), r4(lv)])
    print(ep, r4(lt), r4(lv), flush=True)
    if lv < best - 1e-6: best, bad, sdict = lv, 0, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    else: bad += 1
    if bad >= HP['patience']: break
m.load_state_dict(sdict); m.eval()
with torch.no_grad():
    P = torch.cat([m(Xt[torch.as_tensor(vai[b:b + 4096], device=dev)].float()).cpu() for b in range(0, len(vai), 4096)]).numpy()
with torch.no_grad():
    Qv = torch.cat([m.both(Xt[torch.as_tensor(vai[b:b + 4096], device=dev)].float())[1].cpu() for b in range(0, len(vai), 4096)]).numpy()
Mv = NEED[vai]
def r2(s_, y):
    r = np.argsort(np.argsort(s_)) + 1.0; n1 = y.sum(); n0 = len(y) - n1; return r4((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None
Yv, cx, cr, cc, zv, rv = Y[vai], Cxy[vai], Crot[vai], C[vai], Z[vai], ROT[vai]
v = dict(cos_xy=cosm(P, Yv, cx, slice(0, 2)), cos_rot=cosm(P, Yv, cr, slice(3, 6)),
         nocorr_zone_xy=r4(np.abs(P[zv & ~cc, :2]).mean()), nocorr_zone_rot=r4(np.abs(P[zv & ~cc, 3:6]).mean()),
         far_xy=r4(np.abs(P[~zv, :2]).mean()) if (~zv).any() else None, far_rot=r4(np.abs(P[~zv, 3:6]).mean()) if (~zv).any() else None,
         rotating_rz=r4(np.abs(P[rv > 15, 5]).mean()) if (rv > 15).any() else None,
         mae_xy_corr=r4(np.abs(P[cx, :2] - Yv[cx, :2]).mean()), label_abs_xy_corr=r4(np.abs(Yv[cx, :2]).mean()),
         n_val=int(len(vai)), n_val_zone=int(zv.sum()), n_val_far=int((~zv).sum()), n_val_rotating=int((rv > 15).sum()), n_xy=int(cx.sum()), n_rot=int(cr.sum()))
chk = dict(cos_xy=v['cos_xy'] is not None and v['cos_xy'] >= 0.5, cos_rot=v['cos_rot'] is not None and v['cos_rot'] >= 0.5,
           nocorr_zone=v['nocorr_zone_xy'] <= 0.05 and v['nocorr_zone_rot'] <= 0.05,
           far=v['far_xy'] is not None and v['far_xy'] <= 0.05 and v['far_rot'] <= 0.05,
           rotating=v['rotating_rz'] is not None and v['rotating_rz'] <= 0.05)
ok = all(chk.values()); path = J / ('residual_conv.pt' if not MAXF else 'presmoke_residual.pt')
torch.save({'model': m.state_dict(), 'variant': 'HF_CONV', 'hp': HP, 'criteria_sha256': sha}, path)
res = dict(criteria_sha256=sha, hp=HP, n_rows=N, rows_by_source={s_: int((SRC == s_).sum()) for s_ in np.unique(SRC)}, n_files=len(files),
           n_zone=int(Z.sum()), n_corr_xy=int(Cxy.sum()), n_corr_rot=int(Crot.sum()),
           history=hist, val=v, head_val=dict(auroc_need_xy=r2(Qv[:, 0], Mv[:, 0]), auroc_need_rot=r2(Qv[:, 1], Mv[:, 1]), auroc_need_xy_zone=r2(Qv[zv, 0], Mv[zv, 0]), auroc_need_rot_zone=r2(Qv[zv, 1], Mv[zv, 1])), checks=chk, G_OFF_BASE='PASS' if ok else 'FAIL', ckpt=str(path), ckpt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), robot_command_sent=False)
(J / ('train_summary.json' if not MAXF else 'presmoke_train_summary.json')).write_text(json.dumps(res, indent=1) + '\n')
print(json.dumps({k: v for k, v in res.items() if k != 'history'}))
