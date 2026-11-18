"""q_hist_20261110: + --hist (append history features to P_generic) / --hist_shuffle (control). q_balance_20261110: copy of s13j train_iql.py + --episode_balance (row weight = 1/rows of its (run, env) episode, mean 1 over train rows; task-agnostic, no flip label) applied to both V and Q losses. Original docstring: usb_keyed_s13a_vgps_f15_20261107 (copy of usb_keyed_s8_sel_20261102, ckpt name usb15_iql) (copy of multisocket_vgps_20261031/train_iql.py; changes: 25% of runs held out with seed 20261102,
ckpt names usb_iql_s*.pt, --shuffle_state control = obs_features / P_generic rows permuted across training rows (SHUF), diag on val only):
IQL (Kostrikov et al. 2022) on the refresh-level dataset (build_dataset.py) for the
V-GPS-style baseline (Nakamoto et al. 2024: re-rank a generalist's K samples by an offline-RL Q).

Q = the server's dueling verifier_model.Verifier in rawplan mode (cand = plan_full 12x6 + dist + 1, P = P_generic 28,
no tactile, no align): Q(s, a_k) = v(s) + A_k - mean_k A over the 8 stored candidates.
V = separate MLP on (obs_features, P_generic), expectile tau 0.7 on the target-Q of the executed candidate.
Q target = r + gamma (1 - done) V(s'), gamma 0.97 per refresh (12 steps).
One IQL per seed; every member uses the same normalization stats (the server applies the first member's stats), and
the server deploys the comma-separated members with the pessimistic min over A.
Split: by run (10% val, fixed seed). Fixed number of epochs, last epoch kept (no early stop: the TD loss rises as
value propagates through bootstrapping, so its minimum is the untrained epoch 0 -- seen in dev, 2026-10-31).
"""
import argparse, json, os, sys, time, numpy as np, torch
from torch import nn
sys.path.insert(0, '/home/dyros-recruit/mcy_ws/reactive_diffusion_policy')
from verifier_model import Verifier

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument('--data', default=f'{HERE}/data')
p.add_argument('--out', default=f'{HERE}/ckpt')
p.add_argument('--seeds', default='0,1,2')
p.add_argument('--gamma', type=float, default=0.97)
p.add_argument('--expectile', type=float, default=0.7)
p.add_argument('--epochs', type=int, default=100)
p.add_argument('--batch', type=int, default=1024)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--tau_target', type=float, default=0.005)
p.add_argument('--max_rows', type=int, default=0, help='dev only: subsample rows')
p.add_argument('--shuffle_state', action='store_true', help='SHUF control: permute (obs_features, P_generic) across training rows')
p.add_argument('--name', default='usb15_iql')
p.add_argument('--episode_balance', action='store_true')
p.add_argument('--hist', default='')
p.add_argument('--hist_shuffle', action='store_true')
p.add_argument('--mem_shuffle', action='store_true')
p.add_argument('--runs', default='', help='data budget: comma list of run indices used for training')
args = p.parse_args()
dev = 'cuda:0'
os.makedirs(args.out, exist_ok=True)

L = lambda k: np.load(f'{args.data}/{k}.npy', mmap_mode='r')
run = np.asarray(L('run')); N = len(run)
rng = np.random.default_rng(20261102)
uruns = np.unique(run); val_runs = set(rng.choice(uruns, max(1, len(uruns) // 4), replace=False).tolist())
is_val = np.array([r in val_runs for r in run])
if args.runs:
    is_val = ~np.isin(run, [int(x) for x in args.runs.split(',')])
tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
if args.runs:
    va_idx = va_idx[:256]; args.epochs = int(round(100 * 29140 / len(tr_idx)))
if args.max_rows:
    tr_idx, va_idx = tr_idx[:args.max_rows], va_idx[:max(1, args.max_rows // 10)]

V16 = np.array(L('V')); P = np.array(L('P')); C16 = np.asarray(L('C'))
if args.mem_shuffle: P[:, 28:] = P[np.random.default_rng(20261110).permutation(N), 28:]
if args.hist:
    _H = np.load(args.hist).astype(np.float32); assert len(_H) == N
    if args.hist_shuffle: _H = _H[np.random.default_rng(20261110).permutation(N)]
    P = np.concatenate([P.astype(np.float32), _H], 1)
if args.shuffle_state:
    perm_s = np.random.default_rng(20261103).permutation(tr_idx)
    V16[tr_idx] = V16[perm_s]; P[tr_idx] = P[perm_s]
stats = {'V': (V16[tr_idx].astype(np.float32).mean(0), V16[tr_idx].astype(np.float32).std(0) + 1e-3),
         'P': (P[tr_idx].mean(0), P[tr_idx].std(0) + 1e-6),
         'C': (C16[tr_idx].astype(np.float32).reshape(-1, C16.shape[-1]).mean(0),
               C16[tr_idx].astype(np.float32).reshape(-1, C16.shape[-1]).std(0) + 1e-6)}
g = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)
Vg, Pg, Cg = g(V16), g(P), g(C16)
st = {k: (g(a.astype(np.float32)), g(b.astype(np.float32))) for k, (a, b) in stats.items()}
_env = np.asarray(L('env')); _key = run.astype(np.int64) * 100 + _env; _, _inv, _cnt = np.unique(_key, return_inverse=True, return_counts=True)
W = (1.0 / _cnt[_inv]).astype(np.float32) if args.episode_balance else np.ones(N, np.float32)
W = W / W[tr_idx].mean(); Wg = g(W)
nxt = g(np.asarray(L('next'))); rew = g(np.asarray(L('rew'))); done = g(np.asarray(L('done')))
DC, DP, DV = C16.shape[-1], P.shape[-1], V16.shape[-1]
print('rows', N, 'train', len(tr_idx), 'val', len(va_idx), 'val_runs', len(val_runs), flush=True)


def inputs(b):
    zv = (Vg[b].float() - st['V'][0]) / st['V'][1]
    zp = (Pg[b] - st['P'][0]) / st['P'][1]
    zc = (Cg[b].float() - st['C'][0]) / st['C'][1]
    return zv, zp, zc


class VNet(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.v_in = nn.Sequential(nn.Linear(DV, h), nn.SiLU(), nn.Dropout(0.1))
        self.f = nn.Sequential(nn.Linear(h + DP, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, zv, zp):
        return self.f(torch.cat([self.v_in(zv), zp], 1))[:, 0]


def qfwd(m, zv, zp, zc):
    n = zv.shape[0]
    _, _, Q, A = m(zv, zp, torch.zeros(n, 0, device=dev), torch.zeros(n, device=dev), zc,
                   torch.zeros(n, zc.shape[1], 2, device=dev))
    return Q, A


def losses(q, qt, vnet, b, train=True):
    zv, zp, zc = inputs(b)
    with torch.no_grad():
        q_tgt_exec = qfwd(qt, zv, zp, zc)[0][:, 0]
        nb = nxt[b]; has = nb >= 0
        vn = torch.zeros(len(b), device=dev)
        if has.any():
            zv2, zp2, _ = inputs(nb[has])
            vn[has] = vnet(zv2, zp2)
        y = rew[b] + args.gamma * (1 - done[b]) * vn
    v = vnet(zv, zp)
    u = q_tgt_exec - v
    lv = (Wg[b] * torch.abs(args.expectile - (u < 0).float()) * u ** 2).sum() / Wg[b].sum()
    Q, A = qfwd(q, zv, zp, zc)
    lq = (Wg[b] * (Q[:, 0] - y) ** 2).sum() / Wg[b].sum()
    return lv, lq, Q


summary = {'val_runs': sorted(int(x) for x in val_runs), 'args': vars(args), 'n_rows': int(N), 'n_train': int(len(tr_idx)), 'n_val': int(len(va_idx)), 'members': []}
paths = []
for seed in [int(s) for s in args.seeds.split(',')]:
    torch.manual_seed(seed); np.random.seed(seed)
    q = Verifier(dp=DP, dc=DC, dt=0, use_align=False, use_vision=True).to(dev)
    qt = Verifier(dp=DP, dc=DC, dt=0, use_align=False, use_vision=True).to(dev); qt.load_state_dict(q.state_dict()); qt.eval()
    vnet = VNet().to(dev)
    oq = torch.optim.Adam(q.parameters(), lr=args.lr); ov = torch.optim.Adam(vnet.parameters(), lr=args.lr)
    hist = []
    t0 = time.time()
    for ep in range(args.epochs):
        q.train(); vnet.train(); perm = np.random.permutation(tr_idx); s_lv = s_lq = 0.0; nb_ = 0
        for i in range(0, len(perm), args.batch):
            b = g(perm[i:i + args.batch])
            lv, lq, _ = losses(q, qt, vnet, b)
            ov.zero_grad(); lv.backward(); ov.step()
            oq.zero_grad(); lq.backward(); oq.step()
            with torch.no_grad():
                for a_, b_ in zip(qt.parameters(), q.parameters()):
                    a_.mul_(1 - args.tau_target).add_(args.tau_target * b_)
            s_lv += float(lv); s_lq += float(lq); nb_ += 1
        q.eval(); vnet.eval(); vlv = vlq = 0.0; m_ = 0
        with torch.no_grad():
            for i in range(0, len(va_idx), 4096):
                b = g(va_idx[i:i + 4096]); lv, lq, _ = losses(q, qt, vnet, b, False)
                vlv += float(lv) * len(b); vlq += float(lq) * len(b); m_ += len(b)
        vlv /= m_; vlq /= m_
        hist.append({'epoch': ep, 'train_lv': s_lv / nb_, 'train_lq': s_lq / nb_, 'val_lv': vlv, 'val_lq': vlq})
        print(seed, hist[-1], f'{time.time() - t0:.0f}s', flush=True)
    best, best_ep = vlq, ep
    best_sd = {k: v.detach().clone() for k, v in q.state_dict().items()}
    best_v = {k: v.detach().clone() for k, v in vnet.state_dict().items()}
    path = f'{args.out}/{args.name}_s{seed}.pt'
    torch.save({'state_dict': best_sd, 'vnet_state_dict': best_v, 'tau': 0.0, 'generic': True, 'rawplan': True,
                'dp': DP, 'cand_features': DC, 'dt': 0, 'use_align': False, 'use_vision': True, 'feature_set': None,
                'stats': {k: (np.asarray(a), np.asarray(b)) for k, (a, b) in stats.items()}, 'gamma': args.gamma,
                'expectile': args.expectile, 'last_epoch': best_ep, 'seed': seed, 'doc': __doc__}, path)
    paths.append(path)
    summary['members'].append({'seed': seed, 'last_epoch': best_ep, 'last_val_lq': best, 'history': hist, 'path': path})

# report-only diagnostics on val rows (ensemble = pessimistic min over A, as the server)
members = []
for pth in paths:
    ck = torch.load(pth, map_location=dev, weights_only=False)
    m = Verifier(dp=DP, dc=DC, dt=0, use_align=False, use_vision=True).to(dev); m.load_state_dict(ck['state_dict']); m.eval()
    members.append(m)
win48 = np.asarray(L('win48')); tt = np.asarray(L('t'))
arg, spread, Qm = [], [], []
with torch.no_grad():
    for i in range(0, len(va_idx), 4096):
        b = g(va_idx[i:i + 4096]); zv, zp, zc = inputs(b)
        A = torch.stack([qfwd(m, zv, zp, zc)[1] for m in members]).min(0).values
        arg.append(A.argmax(1).cpu().numpy()); spread.append((A.max(1).values - A.min(1).values).cpu().numpy())
        Qm.append(qfwd(members[0], zv, zp, zc)[0][:, 0].cpu().numpy())
arg = np.concatenate(arg); agree = arg == 0
w = win48[va_idx]
summary['diag_val'] = {
    'argmaxQ_is_executed_frac': float(agree.mean()),
    'win48_when_executed_is_argmax': float(w[agree].mean()) if agree.any() else None,
    'win48_when_executed_not_argmax': float(w[~agree].mean()) if (~agree).any() else None,
    'A_range_over_8_median': float(np.median(np.concatenate(spread))), 'Q_exec_mean': float(np.concatenate(Qm).mean()),
    'note': 'report only; executed candidate is index 0 of 8 (chance 1/8); confounded by the behaviour policy'}
summary['ckpts'] = ','.join(paths)
json.dump(summary, open(f'{args.out}/train_summary_{args.name}.json', 'w'), indent=1)
print(json.dumps(summary['diag_val']), summary['ckpts'], flush=True)
