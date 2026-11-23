"""rr_atnorm_20261123: reconstruction of the FIXED AT (at_20261120) under two normalizers.
N0 = statistics of the 45 demonstrations (what the AT was trained with; taken from ldp_jit_20261121/normalizer.pkl)
N1 = statistics of 45 demonstrations + 27 LEAVE segments (ldp_tkL_20261123/normalizer.pkl, what the running training hands to the AT)
CPU only, no robot, no images. Usage: python atnorm_20261123.py <out.json>"""
import sys, os, glob, json, pickle, copy, numpy as np, torch, dill, zarr, hydra
from omegaconf import OmegaConf
W = os.path.dirname(os.path.abspath(__file__)); D = W + '/data/atnorm_20261123'
sys.path.insert(0, W + '/rdp_5090')
OmegaConf.register_new_resolver("eval", eval, replace=True)
torch.set_num_threads(4)
p = torch.load(W + '/at_20261120/checkpoints/latest.ckpt', pickle_module=dill, map_location='cpu')
cfg = p['cfg']
at = hydra.utils.instantiate(cfg.policy); at.load_state_dict(p['state_dicts']['model']); at.to('cpu'); at.eval()
N0 = pickle.load(open(D + '/norm/ldp_jit_20261121.pkl', 'rb')); N1 = pickle.load(open(D + '/norm/ldp_tkL_20261123.pkl', 'rb'))

def write(path, eps):
    g = zarr.open(path + '/replay_buffer.zarr', 'w'); d = g.create_group('data'); m = g.create_group('meta')
    for k in ['action', 'contact_8d_right', 'state']:
        d.create_dataset(k, data=np.concatenate([e[k] for e in eps]).astype(np.float32), chunks=(4096,) + eps[0][k].shape[1:])
    m.create_dataset('episode_ends', data=np.cumsum([len(e['action']) for e in eps]).astype(np.int64))

z = zarr.open(D + '/tkL_lowdim/replay_buffer.zarr', 'r'); ends = z['meta/episode_ends'][:]; st = np.r_[0, ends[:-1]]
allep = [{k: z['data/' + k][a:b] for k in ['action', 'contact_8d_right', 'state']} for a, b in zip(st, ends)]
def npz(pat):
    out = []
    for f in sorted(glob.glob(D + '/segments/' + pat)):
        d = np.load(f, allow_pickle=False); out.append({k: d[k] for k in ['action', 'contact_8d_right', 'state']})
    return out
groups = {'demo45': allep[:45], 'leave_train27': allep[45:], 'leave_held6': npz('held_LEAVE_*.npz'),
          'align_train': npz('train_ALIGN_*.npz'), 'align_held6': npz('held_ALIGN_*.npz')}
assert len(allep) == 72 and len(groups['leave_held6']) == 6 and len(groups['align_held6']) == 6
# the copied LEAVE training episodes must be the same data as the segment files
lt = npz('train_LEAVE_*.npz'); assert len(lt) == 27 and sum(len(e['action']) for e in lt) == sum(len(e['action']) for e in groups['leave_train27'])

def mix(a_src, t_src):
    n = copy.deepcopy(N0); n['action'] = copy.deepcopy(a_src['action']); n['contact_8d_right'] = copy.deepcopy(t_src['contact_8d_right']); return n
NORMS = {'N0_at': mix(N0, N0), 'N1_tkL': mix(N1, N1), 'actN1_tacN0': mix(N1, N0), 'actN0_tacN1': mix(N0, N1)}
res = {'n_eps': {k: len(v) for k, v in groups.items()}}
dcfg = OmegaConf.to_container(cfg.task.dataset, resolve=True); dcfg['val_ratio'] = 0.0
for g, eps in groups.items():
    path = D + '/zarr_' + g; write(path, eps); dcfg['dataset_path'] = path
    ds = hydra.utils.instantiate(OmegaConf.create(dcfg))
    if g == 'demo45':   # N0 must be what the AT dataset itself fits on the demonstrations
        nn_ = ds.get_normalizer()
        res['N0_check_max_abs_diff'] = {k: float((nn_[k].params_dict['scale'] - N0[k].params_dict['scale']).abs().max()) for k in ['action', 'contact_8d_right']}
    B = [ds[i] for i in range(len(ds))]
    act = torch.stack([b['action'] for b in B]).float(); ext = torch.stack([b['extended_obs']['contact_8d_right'] for b in B]).float()
    r = {'n_chunks': len(B), 'mean_abs_cmd': act.abs().mean(dim=(0, 1)).tolist()}
    n0y = N0['action'].normalize(act)[..., 1]
    r['frac_chunks_y_outside_AT_range'] = float((n0y.abs().amax(dim=1) > 1.0).float().mean()); r['max_abs_norm_y_under_N0'] = float(n0y.abs().max())
    n0t = N0['contact_8d_right'].normalize(ext); r['frac_chunks_tactile_outside_AT_range'] = float((n0t.abs().amax(dim=(1, 2)) > 1.0).float().mean())
    for nm, N in NORMS.items():
        at.set_normalizer(N); torch.manual_seed(0)
        with torch.no_grad():
            s = at.preprocess(at.normalizer['action'].normalize(act) / at.act_scale)
            zq, _ = at.quant_state_without_vq(at.encoder(s)); zq = at.postprocess_quant_state_without_vq(zq)
            rec = at.get_action_from_latent_with_temporal_cond(zq, at.get_temporal_cond({'contact_8d_right': ext}))
            raw = at.normalizer['action'].unnormalize(rec)
        e = (raw - act).abs()
        r[nm] = {'err_all': float(e.mean()), 'err_y': float(e[..., 1].mean()), 'err_per_dim': e.mean(dim=(0, 1)).tolist(),
                 'err_first8_all': float(e[:, :8].mean()), 'err_first8_y': float(e[:, :8, 1].mean()),
                 'p95_chunk_err_y': float(torch.quantile(e[..., 1].mean(dim=1), 0.95))}
    res[g] = r
json.dump(res, open(sys.argv[1], 'w'), indent=1); print(json.dumps(res, indent=1))
