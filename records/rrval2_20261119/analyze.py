import json, numpy as np
O='/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs'; J=f'{O}/rrval2_20261119'; COH=range(76700,76900,25); S=20260921
a,b,ra,rb=[],[],[],[]; first=None
for c in COH:
    x=np.load(f'{J}/eval/RRB_s{S}_c{c}/rollout_traj.npz'); y=np.load(f'{O}/evq_20261117/eval/Q400B_s{S}_c{c}/rollout_traj.npz')
    for z,L,R in ((x,a,ra),(y,b,rb)):
        su=z['success'].astype(bool); L.append(su.any(0)); R.append(np.array([np.flatnonzero(su[:,e])[0] if su[:,e].any() else 600 for e in range(su.shape[1])]))
    T=min(len(x['action_policy']),len(y['action_policy'])); d=np.abs(x['action_policy'][:T]-y['action_policy'][:T]).max(axis=(1,2)); k=np.flatnonzero(d>1e-4)
    if first is None and len(k): first=dict(cohort=c,step=int(k[0]),diff=float(d[k[0]]))
a,b=np.concatenate(a),np.concatenate(b); s,r=int(a.sum()),int(b.sum()); same=int((a==b).sum())
v='REPRODUCED' if (same>=195 and abs(s-r)<=3 and len(a)==200) else 'CLOSE' if abs(s-r)<=6 else 'DIFFERENT'
out=dict(robot_command_sent=False,RRB=dict(scenes=int(len(a)),success=s,reference=r,same_scenes=same,rmst=round(float(np.concatenate(ra).mean()),1),rmst_reference=round(float(np.concatenate(rb).mean()),1),first_command_difference=first,verdict=v))
json.dump(out,open(f'{J}/summary.json','w'),indent=1); print(json.dumps(out))
