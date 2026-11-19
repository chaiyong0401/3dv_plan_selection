"""stallv_20261119: realization ratio + linear Otsu, confirmation on other episodes (prereg.json). Offline, CPU."""
import json, numpy as np
from scipy.stats import rankdata
O='/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs'; J=f'{O}/stallv_20261119'
src=open(f'{O}/scale_rule_20261117/run.py').read(); ns={}; exec(src.split('# theta dev')[0].replace("assert sha ==","_=sha =="),ns); otsu=ns['otsu']
lin=lambda x:otsu(np.minimum(x,np.percentile(x,99)))[0]
COH=list(range(76700,76900,25)); SEEDS=[20260921,20260922,20260923]; W=24
def wsum(x):
    cs=np.cumsum(np.concatenate([np.zeros((1,)+x.shape[1:]),x]),0); out=np.full((x.shape[0]+1,)+x.shape[1:],np.nan); out[W:]=cs[W:]-cs[:-W]; return out
def feats(d):
    c=0.01*d['action_exec'][...,:3].astype(np.float64); p=d['ee_pos'].astype(np.float64); r=p[1:]-p[:-1]; cc=c[1:]; n=np.linalg.norm(cc,axis=-1)
    pi=np.where(n>1e-9,(r*cc).sum(-1)/np.maximum(n,1e-9),0.0); d24=np.full((p.shape[0],)+p.shape[1:2],np.nan); d24[W:]=np.linalg.norm(p[W:]-p[:-W],axis=-1)
    return wsum(pi)/np.maximum(wsum(n),1e-9),d24
def head(q):
    x,y,z,w=(q[...,i] for i in range(4)); h=np.degrees(np.unwrap(np.arctan2(2*(x*y+w*z),1-2*(y*y+z*z)),axis=0)); return np.abs((h-h[0:1]+180)%360-180)
def load(paths):
    E=[]; pool=[]
    for r in paths:
        d=np.load(f'{r}/rollout_traj.npz'); su=d['success'].astype(bool); fl=np.load(f'{r}/initial_state.npz')['flip'].astype(bool); T=su.shape[0]; h=head(d['plug_quat']); rho,d24=feats(d)
        for e in range(su.shape[1]):
            end=int(np.flatnonzero(su[:,e])[0]) if su[:,e].any() else T; r90=np.flatnonzero(h[:,e]>90); w=min(end,int(r90[0]) if len(r90) else T)
            pool.append(rho[W:end,e]); E.append((bool(fl[e]),rho[W:w,e] if w>W else np.zeros(0),d24[W:w,e] if w>W else np.zeros(0)))
    return E,np.concatenate(pool)
S13P=f'{O}/usb_keyed_s13p_role_2x2_20261108/eval'; S13S=f'{O}/usb_keyed_s13s_seed_rep_20261108/eval'
def old(arm,tags): return [f'{S13P}/{arm}_c{c}' if t=='p' else f'{S13S}/{arm}_s{t}_c{c}' for t in tags for c in COH]
def qroot(s): return f'{O}/evq_20261117/eval' if s==20260921 else f'{O}/b2_20261117/eval'
SETS={'FIRST0':old('FIRST0',['p','20261029','20261030']),'FIRSTR':old('FIRSTR',['p','20261029','20261030']),'Q0':old('Q0',['20261029','20261030']),'QR':old('QR',['20261029','20261030']),
 'BLOCK':[f'{O}/belief_parts_20261116/eval/BLOCK_s{s}_c{c}' for s in SEEDS for c in COH],
 'Q400B':[f'{qroot(s)}/Q400B_s{s}_c{c}' for s in SEEDS for c in COH],
 'Q400B0':[f'{O}/resoff_cur_20261117/eval/Q400B0_s{s}_c{c}' if s==20260921 else f'{O}/resoff3_20261119/eval/Q400B0_s{s}_c{c}' for s in SEEDS for c in COH]}
_,devpool=load([f'{O}/unify0_20261116/eval/NORULE_s{s}_c{c}' for s in SEEDS for c in COH]); TH=float(lin(devpool))
D={k:load(v) for k,v in SETS.items()}
def first(E,new):
    out=[]
    for fl,rho,d24 in E:
        i=np.flatnonzero(rho<TH) if new else np.flatnonzero(d24<0.003); out.append(int(i[0])+W if len(i) else None)
    return out
def mt(k,new):
    E=D[k][0]; fl=np.array([x[0] for x in E]); FS=first(E,new); has=np.array([x is not None for x in FS]); sc=np.array([-x if x is not None else -1e9 for x in FS],float); rk=rankdata(sc); n1=fl.sum(); n0=len(fl)-n1
    return dict(n=int(len(fl)),TPR=round(float(has[fl].mean()),3),FPR=round(float(has[~fl].mean()),3),AUROC=round(float((rk[fl].sum()-n1*(n1+1)/2)/(n1*n0)),3))
def delay(k):
    E=D[k][0]; a=first(E,True); b=first(E,False); dl=np.array([x-y for (f,_,_),x,y in zip(E,a,b) if f and x is not None and y is not None],float)
    return dict(n=int(len(dl)),median=float(np.median(dl)),p90=float(np.percentile(dl,90)),mean=round(float(dl.mean()),1))
out=dict(study='stallv_20261119',robot_command_sent=False,threshold_dev=round(TH,4),refit={k:round(float(lin(D[k][1])),4) for k in D},
  mode_test={k:dict(OLD=mt(k,False),NEW=mt(k,True)) for k in D},delay_vs_OLD_flip={k:delay(k) for k in ('FIRSTR','QR','BLOCK','Q400B')})
m=out['mode_test']; ON=('FIRSTR','QR','BLOCK')
P=dict(V1=all(abs(out['refit'][k]-TH)<=0.3*TH for k in ON),
  V2=all(m[k]['NEW']['AUROC']>=m[k]['OLD']['AUROC']-0.02 and m[k]['NEW']['FPR']<=m[k]['OLD']['FPR']+0.03 and m[k]['NEW']['TPR']>=m[k]['OLD']['TPR']-0.05 for k in ON),
  V3=all(m[a]['NEW']['FPR']-m[b]['NEW']['FPR']>=0.30 for a,b in (('FIRST0','FIRSTR'),('Q0','QR'))),
  V4=all(out['delay_vs_OLD_flip'][k]['median']<=12 and out['delay_vs_OLD_flip'][k]['p90']<=36 for k in ON))
out['predictions']={k:bool(v) for k,v in P.items()}; out['V2_per_set']={k:bool(m[k]['NEW']['AUROC']>=m[k]['OLD']['AUROC']-0.02 and m[k]['NEW']['FPR']<=m[k]['OLD']['FPR']+0.03 and m[k]['NEW']['TPR']>=m[k]['OLD']['TPR']-0.05) for k in ON}
out['verdict']='PASS' if all(P.values()) else 'NOT_ADOPTED'
ex={}
for k in D:
    E=D[k][0]; rho=np.concatenate([x[1] for x in E]); d24=np.concatenate([x[2] for x in E]); a=first(E,True); b=first(E,False); nf=[i for i,x in enumerate(E) if not x[0]]
    ex[k]=dict(row_agreement=round(float(((rho<TH)==(d24<0.003)).mean()),4),flag_share_new=round(float((rho<TH).mean()),3),flag_share_old=round(float((d24<0.003).mean()),3),
      nonflip_new_only=int(sum(a[i] is not None and b[i] is None for i in nf)),nonflip_old_only=int(sum(a[i] is None and b[i] is not None for i in nf)),nonflip_n=len(nf))
out['exploratory']=ex
json.dump(out,open(f'{J}/summary.json','w'),indent=1); print(json.dumps(out))
