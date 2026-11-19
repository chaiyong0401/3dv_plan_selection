"""stallabs_20261119 (exploratory, CPU): does the scale_rule_20261117 procedure (Otsu, eta-based scale pick) applied to the ABSOLUTE
24-step hand displacement recover the hand-set 3 mm? Same dev (unify0 NORULE) and holdout (qdata 5 arms) rows as scale_rule."""
import json, numpy as np
O='/home/dyros-recruit/mcy_ws/manifeel/manifeel/data/outputs'
src=open(f'{O}/scale_rule_20261117/run.py').read()
ns={}; exec(src.split('# theta dev')[0].replace("assert sha ==","_=sha ==").replace("J = f'{O}/scale_rule_20261117'","J = f'{O}/scale_rule_20261117'"),ns)
otsu,rule=ns['otsu'],ns['rule']
def d24(paths):
    out=[]
    for r,rec in paths:
        d=np.load(f'{r}/rollout_traj.npz'); su=d['success'].astype(bool); T,N=su.shape; z=np.load(rec); vs=z['verifier_step'].astype(int); P=z['verifier_P']
        for e in range(N):
            end=int(np.flatnonzero(su[:,e])[0]) if su[:,e].any() else T; js=[j for j,t in enumerate(vs) if int(t)-1<end]
            if js: out.append(P[js,e,10])
    return np.concatenate(out)
dev=d24([(f'{O}/unify0_20261116/eval/NORULE_s{s}_c{c}', json.load(open(f'{O}/unify0_20261116/eval/NORULE_s{s}_c{c}/run_metadata.json'))['server_record_path']) for s in (20260921,20260922,20260923) for c in range(76700,76900,25)])
H=f'{O}/qdata_20261116/eval'
hold=d24([(f'{H}/{a}_s20260921_c{c}', f'{H}/server_records_{a}_s20260921_c{c}.npz') for a in ('Q200_OLD','Q400_OLD','Q200_NV','Q400_NV','QFULL_NV') for c in range(76700,76900,25)])
r=lambda x: {k:(round(v*1000,3) if k.startswith('th') or k=='threshold' else v) for k,v in rule(x).items()}
out=dict(study='stallabs_20261119',exploratory=True,robot_command_sent=False,unit='mm (thresholds)',dev=r(dev),holdout=r(hold),
  check_P10_is_m=dict(dev_median_mm=round(float(np.median(dev))*1000,2),dev_p10_mm=round(float(np.percentile(dev,10))*1000,2),share_below_3mm_dev=round(float((dev<0.003).mean()),3),share_below_3mm_hold=round(float((hold<0.003).mean()),3)))
json.dump(out,open('summary.json','w'),indent=1); print(json.dumps(out))
# part 2: recursive application (the rule again on the lower class of the first split) -- no new constant
for name,x in (('dev',dev),('holdout',hold)):
    t1=rule(x)['threshold']; lo=x[x<t1]; out[f'{name}_recursive']=dict(first_split_mm=round(t1*1000,2),second=r(lo))
json.dump(out,open('summary.json','w'),indent=1); print(json.dumps({k:v for k,v in out.items() if 'recursive' in k}))
