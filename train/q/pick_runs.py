import json, sys, numpy as np
m = json.load(open('/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs/usb_keyed_s13g_q_hf2_20261107/data/manifest.json'))['runs']
fi = [r['run_index'] for r in m if r['behaviour'] == 'first']; ra = [r['run_index'] for r in m if r['behaviour'] == 'random']
n, d = int(sys.argv[1]), int(sys.argv[2]); k = n // 25; rg = np.random.default_rng(1000 * n + d)
print(','.join(map(str, sorted(rg.choice(fi, k // 4, replace=False).tolist() + rg.choice(ra, k - k // 4, replace=False).tolist()))))
