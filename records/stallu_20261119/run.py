"""stallu_20261119: stall from unrealized commanded motion (prereg.json). Offline, CPU."""
import json, numpy as np
from scipy.stats import rankdata
O='/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs'; J=f'{O}/stallu_20261119'
src=open(f'{O}/scale_rule_20261117/run.py').read(); ns={}; exec(src.split('# theta dev')[0].replace("assert sha ==","_=sha =="),ns); rule=ns['rule']
COH=list(range(76700,76900,25)); SEEDS=[20260921,20260922,20260923]; W=24
def wsum(x):   # trailing W-sum over steps s-23..s, valid for s>=24 (index 0 of per-step arrays = step 1)
    cs=np.cumsum(np.concatenate([np.zeros((1,)+x.shape[1:]),x]),0); out=np.full((x.shape[0]+1,)+x.shape[1:],np.nan); out[W:]=cs[W:]-cs[:-W]; return out
def feats(d):
    c=0.01*d['action_exec'][...,:3].astype(np.float64); p=d['ee_pos'].astype(np.float64); r=p[1:]-p[:-1]; cc=c[1:]; n=np.linalg.norm(cc,axis=-1)
    pi=np.where(n>1e-9,(r*cc).sum(-1)/np.maximum(n,1e-9),0.0)
    Sn=wsum(n); Sp=wsum(pi); Sc=np.full(Sn.shape,np.nan); csv=np.cumsum(np.concatenate([np.zeros((1,)+cc.shape[1:]),cc]),0); Sc[W:]=np.linalg.norm(csv[W:]-csv[:-W],axis=-1)
    d24=np.full(Sn.shape,np.nan); d24[W:]=np.linalg.norm(p[W:]-p[:-W],axis=-1)
    return dict(RHO=Sp/np.maximum(Sn,1e-9),U=(Sn-Sp)*1000,d24=d24,R1=d24/np.maximum(Sc,1e-6),path=Sn)
def end_of(su):
    T=su.shape[0]; return np.array([int(np.flatnonzero(su[:,e])[0]) if su[:,e].any() else T for e in range(su.shape[1])])
def pool(paths):
    out={'RHO':[],'U':[]}
    for r in paths:
        d=np.load(f'{r}/rollout_traj.npz'); f=feats(d); en=end_of(d['success'].astype(bool))
        for e,w in enumerate(en):
            for k in out: out[k].append(f[k][W:w,e])
    return {k:np.concatenate(v) for k,v in out.items()}
dev=pool([f'{O}/unify0_20261116/eval/NORULE_s{s}_c{c}' for s in SEEDS for c in COH])
hold=pool([f'{O}/qdata_20261116/eval/{a}_s20260921_c{c}' for a in ('Q200_OLD','Q400_OLD','Q200_NV','Q400_NV','QFULL_NV') for c in COH])
fit={k:dict(dev=rule(dev[k]),holdout=rule(hold[k])) for k in ('RHO','U')}; TH={k:fit[k]['dev']['threshold'] for k in fit}
FL={'OLD':lambda f,th=None:f['d24']<0.003,'R1':lambda f,th=None:f['R1']<0.0547,'RHO':lambda f,th=None:f['RHO']<(TH['RHO'] if th is None else th),'U':lambda f,th=None:f['U']>(TH['U'] if th is None else th)}
def head(q):
    x,y,z,w=(q[...,i] for i in range(4)); h=np.degrees(np.unwrap(np.arctan2(2*(x*y+w*z),1-2*(y*y+z*z)),axis=0)); return np.abs((h-h[0:1]+180)%360-180)
def qroot(s): return f'{O}/evq_20261117/eval' if s==20260921 else f'{O}/b2_20261117/eval'
def root(arm,s):
    if arm=='Q400B0': return f'{O}/resoff_cur_20261117/eval' if s==20260921 else f'{O}/resoff3_20261119/eval'
    return qroot(s)
CACHE={}
def eps(arm):
    if arm in CACHE: return CACHE[arm]
    L=[]
    for s in SEEDS:
        for c in COH:
            r=f'{root(arm,s)}/{arm}_s{s}_c{c}'; d=np.load(f'{r}/rollout_traj.npz'); su=d['success'].astype(bool); fl=np.load(f'{r}/initial_state.npz')['flip'].astype(bool); T=su.shape[0]; h=head(d['plug_quat']); f=feats(d); en=end_of(su)
            for e in range(su.shape[1]):
                r90=np.flatnonzero(h[:,e]>90); w=min(int(en[e]),int(r90[0]) if len(r90) else T)
                L.append((bool(fl[e]),{k:v[W:w,e] for k,v in f.items()} if w>W else None))
    CACHE[arm]=L; return L
def first(arm,name,th=None):
    FS=[]
    for fl,f in eps(arm):
        if f is None: FS.append(None); continue
        i=np.flatnonzero(FL[name](f,th)); FS.append(int(i[0])+W if len(i) else None)
    return np.array([x[0] for x in eps(arm)]),FS
def mt(arm,name,th=None):
    fl,FS=first(arm,name,th); has=np.array([x is not None for x in FS]); sc=np.array([-x if x is not None else -1e9 for x in FS],float); rk=rankdata(sc); n1=fl.sum(); n0=len(fl)-n1
    return dict(TPR=round(float(has[fl].mean()),3),FPR=round(float(has[~fl].mean()),3),AUROC=round(float((rk[fl].sum()-n1*(n1+1)/2)/(n1*n0)),3))
def delay(name,th=None):
    fl,a=first('Q400B',name,th); _,b=first('Q400B','OLD'); dl=np.array([x-y for f,x,y in zip(fl,a,b) if f and x is not None and y is not None],float)
    return dict(n=int(len(dl)),median=float(np.median(dl)) if len(dl) else None,p90=float(np.percentile(dl,90)) if len(dl) else None,mean=round(float(dl.mean()),1) if len(dl) else None)
out=dict(study='stallu_20261119',robot_command_sent=False,thresholds=fit,mode_test={f'{a}_{n}':mt(a,n) for a in ('Q400B','Q400B0') for n in FL},delay_vs_OLD_flip={n:delay(n) for n in ('R1','RHO','U')})
m=out['mode_test']; P={}
for k in ('RHO','U'):
    d,h=fit[k]['dev'],fit[k]['holdout']; dl=out['delay_vs_OLD_flip'][k]
    P[k]=dict(T1=bool(abs(h['threshold']-d['threshold'])<=0.3*abs(d['threshold']) and d['pick']==h['pick']),
      T2=bool(m[f'Q400B_{k}']['AUROC']>=m['Q400B_OLD']['AUROC']-0.02 and m[f'Q400B_{k}']['FPR']<=m['Q400B_OLD']['FPR']+0.03 and m[f'Q400B_{k}']['TPR']>=m['Q400B_OLD']['TPR']-0.05),
      T3=bool(m[f'Q400B0_{k}']['FPR']-m[f'Q400B_{k}']['FPR']>=0.30),
      T4=bool(dl['n']>0 and dl['median']<=12 and dl['p90']<=36))
out['predictions']=P; out['verdict']={k:('PASS' if all(v.values()) else 'NOT_ADOPTED') for k,v in P.items()}
# exploratory
ex={}
allf={k:np.concatenate([f[k] for _,f in eps('Q400B') if f is not None]) for k in ('RHO','U','d24','R1','path')}
old=allf['d24']<0.003
ex['row_agreement_with_OLD_Q400B_modewindow']={k:round(float(((FL[k](allf))==old).mean()),4) for k in ('R1','RHO','U')}
ex['flag_share']={k:round(float(FL[k](allf).mean()),3) for k in FL}
ex['OLD_flags_with_commanded_path_below_3mm']=round(float((allf['path'][old]<0.003).mean()),4)
ex['commanded_path_mm_in_OLD_flags_quantiles']=[round(float(v*1000),2) for v in np.percentile(allf['path'][old],[5,25,50,75,95])]
ex['RHO_quantiles_dev']=[round(float(v),3) for v in np.percentile(dev['RHO'],[1,5,25,50,75,95,99])]; ex['U_mm_quantiles_dev']=[round(float(v),2) for v in np.percentile(dev['U'],[1,5,25,50,75,95,99])]
fm={}
for k,grid in (('RHO',np.linspace(0.005,0.45,90)),('U',np.linspace(2,80,157))):
    best=min(grid,key=lambda t:abs(mt('Q400B',k,t)['FPR']-m['Q400B_OLD']['FPR'])); fm[k]=dict(threshold=round(float(best),4),on=mt('Q400B',k,best),off=mt('Q400B0',k,best),delay=delay(k,best))
ex['fpr_matched_posthoc']=fm; out['exploratory']=ex
json.dump(out,open(f'{J}/summary.json','w'),indent=1); print(json.dumps(out))
