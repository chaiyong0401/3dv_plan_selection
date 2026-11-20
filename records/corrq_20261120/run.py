"""corrq_20261120: mode test and mediation across correctors (prereg.json). Offline, CPU."""
import json, numpy as np
from scipy.stats import rankdata, spearmanr
O='/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs'; J=f'{O}/corrq_20261120'; COH=[76700+25*i for i in range(8)]; SEEDS=[20260921,20260922,20260923]
def head(q):
    x,y,z,w=(q[...,i] for i in range(4)); h=np.degrees(np.unwrap(np.arctan2(2*(x*y+w*z),1-2*(y*y+z*z)),axis=0)); return np.abs((h-h[0:1]+180)%360-180)
ROOT={'OFF':lambda s:(f'{O}/resoff_cur_20261117/eval' if s==20260921 else f'{O}/resoff3_20261119/eval','Q400B0'),
 'RES':lambda s:(f'{O}/evq_20261117/eval' if s==20260921 else f'{O}/b2_20261117/eval','Q400B'),
 'B2':lambda s:(f'{O}/b2_20261117/eval','Q400B2'),'ER3':lambda s:(f'{O}/errsim3_20261118/eval' if s==20260921 else f'{O}/errsim3x_20261118/eval','Q400ER3'),
 'DZ':lambda s:(f'{O}/dz_20261117/eval','Q400DZ'),
 'RZ0':lambda s:(f'{O}/rzoff_20261118/eval','Q400RZ0'),'LS':lambda s:(f'{O}/q400ls_20261117/eval','Q400LS'),'LINSGN':lambda s:(f'{O}/linres_20261117/eval','LINSGN'),'S25':lambda s:(f'{O}/stallmm_20261119/eval','Q400S25')}
def load(k,seeds=SEEDS,th=0.003):
    F,H,FS,OK,RM=[],[],[],[],[]
    for s in seeds:
        for c in COH:
            ro,arm=ROOT[k](s); r=f'{ro}/{arm}_s{s}_c{c}'; d=np.load(f'{r}/rollout_traj.npz'); su=d['success'].astype(bool); fl=np.load(f'{r}/initial_state.npz')['flip'].astype(bool); T=su.shape[0]; h=head(d['plug_quat']); ee=d['ee_pos']
            for e in range(su.shape[1]):
                ok=bool(su[:,e].any()); end=int(np.flatnonzero(su[:,e])[0]) if ok else T; r90=np.flatnonzero(h[:,e]>90); w=min(end,int(r90[0]) if len(r90) else T)
                disp=np.linalg.norm(ee[24:w,e]-ee[:max(w-24,0),e],axis=1) if w>24 else np.zeros(0); st=np.flatnonzero(disp<th)
                F.append(bool(fl[e])); H.append(len(st)>0); FS.append(-(int(st[0])+24) if len(st) else -1e9); OK.append(ok); RM.append(end if ok else 600)
    return dict(fl=np.array(F),has=np.array(H),sc=np.array(FS,float),ok=np.array(OK),rm=np.array(RM,float))
def mt(D):
    fl=D['fl']; r=rankdata(D['sc']); n1=fl.sum(); n0=len(fl)-n1; nf=~fl
    return dict(n=int(len(fl)),TPR=round(float(D['has'][fl].mean()),3),FPR=round(float(D['has'][nf].mean()),3),AUROC=round(float((r[fl].sum()-n1*(n1+1)/2)/(n1*n0)),3),
      success=int(D['ok'].sum()),succ_flip=int(D['ok'][fl].sum()),succ_nonflip=int(D['ok'][nf].sum()),n_nonflip=int(nf.sum()),rmst=round(float(D['rm'].mean()),1),
      nonflip_succ_given_nostall=[int(D['ok'][nf&~D['has']].sum()),int((nf&~D['has']).sum())],nonflip_succ_given_stall=[int(D['ok'][nf&D['has']].sum()),int((nf&D['has']).sum())])
P=['OFF','RES','B2','ER3','DZ']; D={k:load(k) for k in P}; M={k:mt(D[k]) for k in P}
fl0=D['OFF']['fl']; assert all((D[k]['fl']==fl0).all() for k in P)
def med(a,pool):
    s1=sum(M[k]['nonflip_succ_given_nostall'][0] for k in pool); n1=sum(M[k]['nonflip_succ_given_nostall'][1] for k in pool); s0=sum(M[k]['nonflip_succ_given_stall'][0] for k in pool); n0=sum(M[k]['nonflip_succ_given_stall'][1] for k in pool)
    p1,p0=s1/n1,s0/n0; pred=p1*(1-M[a]['FPR'])+p0*M[a]['FPR']; act=M[a]['succ_nonflip']/M[a]['n_nonflip']; return dict(p1=round(p1,3),p0=round(p0,3),predicted=round(pred,3),actual=round(act,3),error=round(pred-act,3))
ON=P[1:]; MED={a:med(a,[k for k in P if k!=a]) for a in ON}; MED['OFF']=med('OFF',ON)
G=dict(G1=all(M[k]['TPR']>=0.95 for k in P),G2=all(M['OFF']['FPR']-M[k]['FPR']>=0.30 and M[k]['AUROC']>=M['OFF']['AUROC']+0.15 for k in ON),
  G3=all(abs(MED[k]['error'])<=0.05 for k in ON),G4=abs(MED['OFF']['error'])<=0.05)
out=dict(study='corrq_20261120',robot_command_sent=False,mode_test=M,mediation=MED,predictions={k:bool(v) for k,v in G.items()},
  G5_spearman_auroc_success=round(float(spearmanr([M[k]['AUROC'] for k in P],[M[k]['success'] for k in P]).correlation),3))
out['verdict']='SUPPORTED' if all(G.values()) else ('PARTIAL' if G['G1'] and G['G2'] else 'NOT_SUPPORTED')
ex={}
for k,sd in (('RZ0',[20260921]),('LS',[20260921]),('LINSGN',SEEDS),('S25',SEEDS)):
    try: ex[k]=mt(load(k,sd,0.0025 if k=='S25' else 0.003))
    except Exception as e: ex[k]=f'unavailable: {type(e).__name__}'
out['exploratory']=ex; out['per_seed_FPR']={k:[round(float(load(k,[s])['has'][~load(k,[s])['fl']].mean()),3) for s in SEEDS] for k in P}
json.dump(out,open(f'{J}/summary.json','w'),indent=1); print(json.dumps(out))
