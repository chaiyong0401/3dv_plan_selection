"""stallc_20261119: contact-conditioned Otsu stall threshold (prereg.json). Offline, CPU."""
import json, numpy as np
from scipy.stats import rankdata
O='/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs'; J=f'{O}/stallc_20261119'
src=open(f'{O}/scale_rule_20261117/run.py').read(); ns={}; exec(src.split('# theta dev')[0].replace("assert sha ==","_=sha =="),ns); rule=ns['rule']
COH=list(range(76700,76900,25)); SEEDS=[20260921,20260922,20260923]
def gstep(d):   # (T,N) per-step tactile force deviation from episode start
    c=d['contact_8d_right'][...,:3].astype(np.float64); return np.linalg.norm(c-c[0:1],axis=-1)
def gwin(g):    # 24-step trailing mean, index t = mean of g[t-23..t]
    cs=np.cumsum(np.concatenate([np.zeros((1,)+g.shape[1:]),g]),0); out=np.full(g.shape,np.nan); out[23:]=(cs[24:]-cs[:-24])/24; return out
def rows(paths):
    D,G,E=[],[],[]
    for k,(r,rec) in enumerate(paths):
        d=np.load(f'{r}/rollout_traj.npz'); su=d['success'].astype(bool); T,N=su.shape; z=np.load(rec); vs=z['verifier_step'].astype(int); P=z['verifier_P']; gw=gwin(gstep(d))
        for e in range(N):
            end=int(np.flatnonzero(su[:,e])[0]) if su[:,e].any() else T
            for j,t in enumerate(vs):
                s=int(t)-1
                if s<end and s>=23: D.append(P[j,e,10]); G.append(gw[s,e]); E.append((k,e,j))
    return np.array(D,np.float64),np.array(G,np.float64),E
def rec(r): return json.load(open(f'{r}/run_metadata.json'))['server_record_path']
def fit(D,G):
    rc=rule(G); cm=G>rc['threshold']; rs=rule(D[cm]); return rc,rs,cm
mm=lambda r:{k:(round(v*1000,3) if k.startswith('th') else v) for k,v in r.items()}
dev=rows([(p,rec(p)) for s in SEEDS for c in COH for p in [f'{O}/unify0_20261116/eval/NORULE_s{s}_c{c}']])
H=f'{O}/qdata_20261116/eval'
hold=rows([(f'{H}/{a}_s20260921_c{c}',f'{H}/server_records_{a}_s20260921_c{c}.npz') for a in ('Q200_OLD','Q400_OLD','Q200_NV','Q400_NV','QFULL_NV') for c in COH])
rcd,rsd,cmd=fit(dev[0],dev[1]); rch,rsh,cmh=fit(hold[0],hold[1]); THC,THS=rcd['threshold'],rsd['threshold']
out=dict(study='stallc_20261119',robot_command_sent=False,
  dev=dict(contact=rcd,contact_share=round(float(cmd.mean()),3),stall=mm(rsd),all_rows_reference=mm(rule(dev[0]))),
  holdout=dict(contact=rch,contact_share=round(float(cmh.mean()),3),stall=mm(rsh),all_rows_reference=mm(rule(hold[0]))))
# agreement on Q400B
def qroot(s): return f'{O}/evq_20261117/eval' if s==20260921 else f'{O}/b2_20261117/eval'
ag=rows([(p,rec(p)) for s in SEEDS for c in COH for p in [f'{qroot(s)}/Q400B_s{s}_c{c}']])
old=ag[0]<0.003; new=(ag[1]>THC)&(ag[0]<THS)
def consec(f,E):
    o=np.zeros(len(f),bool); prev=None
    for i,(k,e,j) in enumerate(E):
        o[i]=f[i] and prev==(k,e,j-1) and f[i-1]; prev=(k,e,j)
    return o
out['agreement_Q400B']=dict(rows=int(len(old)),old_share=round(float(old.mean()),3),new_share=round(float(new.mean()),3),flag=round(float((old==new).mean()),4),
  old_only=int((old&~new).sum()),new_only=int((new&~old).sum()),flag_M2=round(float((consec(old,ag[2])==consec(new,ag[2])).mean()),4))
# mode test (resoff3 definition)
def head(q):
    x,y,z,w=(q[...,i] for i in range(4)); h=np.degrees(np.unwrap(np.arctan2(2*(x*y+w*z),1-2*(y*y+z*z)),axis=0)); return np.abs((h-h[0:1]+180)%360-180)
def root(arm,s):
    if arm=='Q400B0': return f'{O}/resoff_cur_20261117/eval' if s==20260921 else f'{O}/resoff3_20261119/eval'
    return qroot(s)
def modetest(arm,newc):
    F,FS=[],[]
    for s in SEEDS:
        for c in COH:
            r=f'{root(arm,s)}/{arm}_s{s}_c{c}'; d=np.load(f'{r}/rollout_traj.npz'); su=d['success'].astype(bool); fl=np.load(f'{r}/initial_state.npz')['flip'].astype(bool); T=su.shape[0]; h=head(d['plug_quat']); ee=d['ee_pos']; gw=gwin(gstep(d))
            for e in range(su.shape[1]):
                end=int(np.flatnonzero(su[:,e])[0]) if su[:,e].any() else T; r90=np.flatnonzero(h[:,e]>90); w=min(end,int(r90[0]) if len(r90) else T)
                if w>24:
                    disp=np.linalg.norm(ee[24:w,e]-ee[:w-24,e],axis=1); st=(disp<THS)&(gw[24:w,e]>THC) if newc else disp<0.003; st=np.flatnonzero(st)
                else: st=[]
                FS.append(int(st[0])+24 if len(st) else None); F.append(bool(fl[e]))
    fl=np.array(F); has=np.array([x is not None for x in FS]); sc=np.array([-x if x is not None else -1e9 for x in FS],float); rk=rankdata(sc); n1=fl.sum(); n0=len(fl)-n1
    return dict(TPR=round(float(has[fl].mean()),3),FPR=round(float(has[~fl].mean()),3),AUROC=round(float((rk[fl].sum()-n1*(n1+1)/2)/(n1*n0)),3))
out['mode_test']={f'{a}_{n}':modetest(a,k) for a in ('Q400B','Q400B0') for n,k in (('old',False),('new',True))}
m=out['mode_test']; a=out['agreement_Q400B']; ts=out['dev']['stall']['threshold']*1000; th=out['holdout']['stall']['threshold']*1000
P=dict(S1=2.0<=ts<=4.5,S2=abs(th-ts)<=0.3*ts and out['dev']['stall']['pick']==out['holdout']['stall']['pick'],S3=a['flag']>=0.95,
  S4=m['Q400B_new']['AUROC']>=m['Q400B_old']['AUROC']-0.02 and m['Q400B_new']['FPR']<=m['Q400B_old']['FPR']+0.03 and m['Q400B_new']['TPR']>=m['Q400B_old']['TPR']-0.05,
  S5=m['Q400B0_new']['FPR']-m['Q400B_new']['FPR']>=0.30)
out['stall_threshold_mm']=dict(dev=round(ts,3),holdout=round(th,3)); out['predictions']={k:bool(v) for k,v in P.items()}
out['verdict']='SAME_DECISIONS' if all(P.values()) else ('ADOPTABLE_NEEDS_RERUN' if P['S1'] and P['S2'] and P['S4'] and P['S5'] else 'NOT_ADOPTED')
json.dump(out,open(f'{J}/summary.json','w'),indent=1); print(json.dumps(out))
