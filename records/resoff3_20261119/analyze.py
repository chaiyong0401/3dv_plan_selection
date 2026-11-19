"""resoff3_20261119: corrector off vs on, seeds 21-23, 600 paired eps (prereg.json)."""
import json, numpy as np
from scipy.stats import rankdata
O='/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs'; J=f'{O}/resoff3_20261119'; COH=[76700+25*i for i in range(8)]; SEEDS=[20260921,20260922,20260923]; rng=np.random.default_rng(0)
def head(q):
    x,y,z,w=(q[...,i] for i in range(4)); h=np.degrees(np.unwrap(np.arctan2(2*(x*y+w*z),1-2*(y*y+z*z)),axis=0)); return np.abs((h-h[0:1]+180)%360-180)
def root(arm,s):
    if arm=='Q400B0': return f'{O}/resoff_cur_20261117/eval' if s==20260921 else f'{J}/eval'
    return f'{O}/evq_20261117/eval' if s==20260921 else f'{O}/b2_20261117/eval'
def load(arm):
    F,FS,OK,RM,SD=[],[],[],[],[]
    for s in SEEDS:
        for c in COH:
            r=f'{root(arm,s)}/{arm}_s{s}_c{c}'; d=np.load(f'{r}/rollout_traj.npz'); su=d['success'].astype(bool); fl=np.load(f'{r}/initial_state.npz')['flip'].astype(bool); T=su.shape[0]; h=head(d['plug_quat']); ee=d['ee_pos']
            for e in range(su.shape[1]):
                ok=bool(su[:,e].any()); end=int(np.flatnonzero(su[:,e])[0]) if ok else T
                r90=np.flatnonzero(h[:,e]>90); w=min(end,int(r90[0]) if len(r90) else T)
                disp=np.linalg.norm(ee[24:w,e]-ee[:max(w-24,0),e],axis=1) if w>24 else np.zeros(0)
                st=np.flatnonzero(disp<0.003); FS.append(int(st[0])+24 if len(st) else None); F.append(bool(fl[e])); OK.append(ok); RM.append(end if ok else 600); SD.append(s)
    return dict(fl=np.array(F),fs=FS,ok=np.array(OK,float),rm=np.array(RM,float),sd=np.array(SD))
def stats(D,m=None):
    m=np.ones(len(D['fl']),bool) if m is None else m; fl=D['fl'][m]; fs=[x for x,k in zip(D['fs'],m) if k]
    has=np.array([x is not None for x in fs]); sc=np.array([-x if x is not None else -1e9 for x in fs],float); r=rankdata(sc); n1=fl.sum(); n0=len(fl)-n1
    return dict(n=int(len(fl)),TPR=round(float(has[fl].mean()),3),FPR=round(float(has[~fl].mean()),3),AUROC=round(float((r[fl].sum()-n1*(n1+1)/2)/(n1*n0)),3),
                success=int(D['ok'][m].sum()),success_flip=int(D['ok'][m][fl].sum()),success_nonflip=int(D['ok'][m][~fl].sum()),rmst=round(float(D['rm'][m].mean()),1))
def ci(x):
    I=rng.integers(0,len(x),(4000,len(x))); mm=x[I].mean(1); return [round(float(x.mean()),4),round(float(np.percentile(mm,2.5)),4),round(float(np.percentile(mm,97.5)),4)]
off=load('Q400B0'); on=load('Q400B'); assert (off['fl']==on['fl']).all() and (off['sd']==on['sd']).all()
fl=on['fl']; hA=np.array([x is not None for x in on['fs']]).astype(float); hB=np.array([x is not None for x in off['fs']]).astype(float)
res=dict(study='resoff3_20261119',robot_command_sent=False,off=stats(off),on=stats(on),
  per_seed={str(s):dict(off=stats(off,off['sd']==s),on=stats(on,on['sd']==s)) for s in SEEDS},
  FPR_on_minus_off=ci((hA-hB)[~fl]),success_on_minus_off=ci(on['ok']-off['ok']),rmst_on_minus_off=ci(on['rm']-off['rm']))
a,b=res['on'],res['off']; ps=res['per_seed']
res['predictions']={k:bool(v) for k,v in dict(R1=(a['FPR']<=b['FPR']-0.10) and all(ps[s]['on']['FPR']<ps[s]['off']['FPR'] for s in ps),R2=a['TPR']>=b['TPR']-0.05,R3=a['AUROC']>=b['AUROC']+0.05,R4=res['rmst_on_minus_off'][2]<0,R5=res['success_on_minus_off'][1]>0).items()}
json.dump(res,open(f'{J}/summary.json','w'),indent=1); print(json.dumps(res))
