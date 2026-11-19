import json, numpy as np
O='/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs'; J=f'{O}/stallmm_20261119'; COH=range(76700,76900,25); SEEDS=[20260921,20260922,20260923]
def ref(s): return f'{O}/evq_20261117/eval' if s==20260921 else f'{O}/b2_20261117/eval'
def load(root,arm,seeds):
    S,R,F=[],[],[]
    for s in seeds:
        for c in COH:
            r=f'{root(s) if callable(root) else root}/{arm}_s{s}_c{c}'; su=np.load(f'{r}/rollout_traj.npz')['success'].astype(bool); fl=np.load(f'{r}/initial_state.npz')['flip'].astype(bool)
            S.append(su.any(0)); R.append(np.array([np.flatnonzero(su[:,e])[0] if su[:,e].any() else 600 for e in range(su.shape[1])])); F.append(fl)
    return np.concatenate(S),np.concatenate(R).astype(float),np.concatenate(F)
def cmp(a,b,n):
    rng=np.random.default_rng(0); I=rng.integers(0,n,(5000,n)); ds=(a[0].astype(float)-b[0])[I].mean(1)*100; dr=(a[1]-b[1])[I].mean(1)
    return dict(scenes=int(n),success=int(a[0].sum()),reference=int(b[0].sum()),diff_pp=round(float((a[0].mean()-b[0].mean())*100),2),diff_pp_ci=[round(float(np.percentile(ds,2.5)),2),round(float(np.percentile(ds,97.5)),2)],
        rmst=round(float(a[1].mean()),1),rmst_reference=round(float(b[1].mean()),1),rmst_diff=round(float((a[1]-b[1]).mean()),1),rmst_diff_ci=[round(float(np.percentile(dr,2.5)),1),round(float(np.percentile(dr,97.5)),1)],
        same_scenes=int((a[0]==b[0]).sum()),flip=[int(a[0][a[2]].sum()),int(b[0][b[2]].sum()),int(a[2].sum())],nonflip=[int(a[0][~a[2]].sum()),int(b[0][~b[2]].sum()),int((~a[2]).sum())])
A=load(f'{J}/eval','Q400S25',SEEDS); B=load(ref,'Q400B',SEEDS); assert (A[2]==B[2]).all()
p=cmp(A,B,len(A[0])); p['per_seed']=[[int(A[0][i*200:(i+1)*200].sum()),int(B[0][i*200:(i+1)*200].sum())] for i in range(3)]
lo,hi=p['diff_pp_ci']; rl,rh=p['rmst_diff_ci']; eq=lo>=-5 and hi<=5 and rl>=-25 and rh<=25
p['verdict']='EQUIVALENT' if eq else 'WORSE' if (hi<0 or rl>0) else 'BETTER' if (lo>0 or rh<0) else 'INCONCLUSIVE'
out=dict(study='stallmm_20261119',robot_command_sent=False,Q400S25=p,predictions=dict(M1=p['verdict']=='EQUIVALENT',M2=abs(p['success']-475)<=15))
try:
    C=load(f'{J}/eval','Q400S40',SEEDS[:1]); B1=tuple(x[:200] for x in B); s=cmp(C,B1,200); out['Q400S40_seed21']=s; out['predictions']['M3']=abs(s['success']-s['reference'])<=10
except Exception as e: out['Q400S40_seed21']=dict(error=str(e)[:200])
json.dump(out,open(f'{J}/summary.json','w'),indent=1); print(json.dumps(out))
