"""errsim2_20261118 train (prereg.json): two SEPARATE nets on the same inputs as residual_conv.pt (hand-frame xy slots, wrench history blinded).
  errnet : e_hat = 15*tanh, MSE vs privileged yaw error to the nearer 180 deg (deg); rows = zone & |err| <= 15 (= errhead_20261118 model B recipe)
  gatenet: logit, BCE vs 'teacher rotation active' (any nonzero teacher rotation label), all rows, sample_w weighted
Gate threshold = the logit at which recall on VALIDATION positives (training-source group split) is 0.90 (eval data not used).
Then the offline gate E0 on Q400B eval rows."""
import glob, json, os, sys, time, numpy as np, torch, torch.nn as nn
M='/home/dyros-recruit/mcy_ws/manifeel/manifeel'; sys.path.insert(0,M); O=f'{M}/data/outputs'; J=f'{O}/errsim2_20261118'
from delta_residual_handframe_20261106 import to_hand
torch.manual_seed(0); dev='cuda:0' if torch.cuda.is_available() else 'cpu'; torch.set_num_threads(12)
yaw=lambda q: np.degrees(np.arctan2(2*(q[...,3]*q[...,2]+q[...,0]*q[...,1]),1-2*(q[...,1]**2+q[...,2]**2)))
w=lambda a:(a+180)%360-180; fo=lambda a: w(a-180*np.round(w(a)/180))
def rows(f):
    z=np.load(f); r=np.load(os.path.dirname(f)+'/rollout_traj.npz'); T=r['plug_quat'].shape[0]; ti=np.clip(z['t']-1,0,T-1); e=z['e']
    priv=fo(yaw(r['plug_quat'])-yaw(r['socket_quat']))[ti,e]
    ld=to_hand(z['ld'].astype(np.float32))
    X=np.concatenate([ld,z['img'].astype(np.float32),z['ff'].astype(np.float32)],1).astype(np.float16)
    return dict(X=X,priv=priv,gate=np.abs(z['delta_teacher'][:,3:6]).sum(1)>0,sw=z['sample_w'].astype(np.float32),zone=z['zone'].astype(bool),
                ab=z['ab'][:,5],te=z['delta_teacher'][:,5],st=z['delta'][:,5],e=e)
class OneNet(nn.Module):
    def __init__(s,d,mu,sd,kind,h=256):
        super().__init__(); s.kind=kind; s.register_buffer('mu',torch.as_tensor(mu)); s.register_buffer('sd',torch.as_tensor(sd))
        s.net=nn.Sequential(nn.Linear(d,h),nn.LayerNorm(h),nn.ReLU(),nn.Linear(h,h),nn.LayerNorm(h),nn.ReLU(),nn.Linear(h,h),nn.LayerNorm(h),nn.ReLU(),nn.Linear(h,1))
    def forward(s,x):
        o=s.net((x-s.mu)/s.sd)[:,0]; return 15*torch.tanh(o) if s.kind=='err' else o
if __name__=='__main__':
    files=sorted(glob.glob(f'{O}/usb_keyed_s12r_gatefree_res_20261106/runs/COL_c*/delta_log.npz'))
    for S in ('usb_keyed_s13a_vgps_f15_20261107','usb_keyed_s13d_mcq_more_20261107','usb_keyed_s13e_mcq_t600_20261107'):
        files+=[f for f in sorted(glob.glob(f'{O}/{S}/runs/*_c*/delta_log.npz')) if os.path.exists(os.path.join(os.path.dirname(f),'eval_log.json'))]
    MAXF=int(os.environ.get('MAXF','0')); MAXEP=int(os.environ.get('MAXEP','20'))
    if MAXF: files=files[:MAXF]+files[-MAXF:]
    TR={k:[] for k in ('X','priv','gate','sw','zone','G')}
    for gi,f in enumerate(files):
        d=rows(f)
        for k in ('X','priv','gate','sw','zone'): TR[k].append(d[k])
        TR['G'].append(gi*1000+d['e'])
    TR={k:np.concatenate(v) for k,v in TR.items()}; print('rows',len(TR['G']),flush=True)
    ug=np.unique(TR['G']); rng=np.random.default_rng(0); rng.shuffle(ug); VA=np.isin(TR['G'],list(set(ug[:int(len(ug)*0.2)])))
    mu=TR['X'][~VA][::5].astype(np.float32).mean(0); sd=TR['X'][~VA][::5].astype(np.float32).std(0)+1e-3; B=list(range(21,93))+[99]; mu[B]=0; sd[B]=1e9
    def fit(kind,sel,y,wt):
        net=OneNet(TR['X'].shape[1],mu.astype(np.float32),sd.astype(np.float32),kind).to(dev); opt=torch.optim.Adam(net.parameters(),3e-4,weight_decay=1e-4)
        tri,vai=np.flatnonzero(sel&~VA),np.flatnonzero(sel&VA); best=(1e9,None,-1); bad=0; hist=[]
        def run(idx,train):
            net.train(train); tot=ws=0.0
            for b in range(0,len(idx),256 if train else 8192):
                k=idx[b:b+(256 if train else 8192)]; x=torch.from_numpy(TR['X'][k].astype(np.float32)).to(dev); t=torch.from_numpy(y[k]).to(dev); ww=torch.from_numpy(wt[k]).to(dev)
                with torch.set_grad_enabled(train):
                    p=net(x); l=((((p-t)**2) if kind=='err' else nn.functional.binary_cross_entropy_with_logits(p,t,reduction='none'))*ww).sum()/ww.sum()
                    if train: opt.zero_grad(); l.backward(); opt.step()
                tot+=float(l)*float(ww.sum()); ws+=float(ww.sum())
            return tot/ws
        for ep in range(MAXEP):
            t0=time.time(); lt=run(np.random.default_rng(ep).permutation(tri),True); lv=run(vai,False); hist.append([ep,round(lt,4),round(lv,4),round(time.time()-t0)]); print(kind,hist[-1],flush=True)
            if lv<best[0]: best=(lv,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},ep); bad=0
            else:
                bad+=1
                if bad>=4: break
        net.load_state_dict(best[1]); net.eval(); return net,dict(best_epoch=best[2],hist=hist),vai
    errnet,ie,_=fit('err',TR['zone']&(np.abs(TR['priv'])<=15),np.clip(TR['priv'],-15,15).astype(np.float32),np.ones(len(TR['G']),np.float32))
    gatenet,ig,gva=fit('gate',np.ones(len(TR['G']),bool),TR['gate'].astype(np.float32),TR['sw'])
    with torch.no_grad(): gl=np.concatenate([gatenet(torch.from_numpy(TR['X'][gva[b:b+8192]].astype(np.float32)).to(dev)).cpu().numpy() for b in range(0,len(gva),8192)])
    pos=TR['gate'][gva]; thr=float(np.quantile(gl[pos],0.10)); fpr_val=float(np.mean(gl[~pos]>thr))
    d=TR['X'].shape[1]; del TR
    torch.save(dict(err=errnet.state_dict(),gate=gatenet.state_dict(),d=d,thr=thr,info=dict(err=ie,gate=ig,val_recall=0.9,val_fpr=round(fpr_val,4))),f'{J}/errnet2.pt')
    src=lambda s:f'{O}/evq_20261117/eval' if s==20260921 else f'{O}/b2_20261117/eval'; A={k:[] for k in ('err','ab','st','eh','g')}
    for s in (20260921,20260922,20260923):
        for c in [76700+25*i for i in range(8)]:
            r=rows(f'{src(s)}/Q400B_s{s}_c{c}/delta_log.npz'); m=r['zone']&(np.abs(r['te'])>1e-6)&(np.abs(r['priv'])<=5)
            with torch.no_grad():
                x=torch.from_numpy(r['X'][m].astype(np.float32)).to(dev); eh=errnet(x).cpu().numpy(); g=gatenet(x).cpu().numpy()
            for k,v in (('err',r['priv'][m]),('ab',r['ab'][m]),('st',r['st'][m]),('eh',eh),('g',g)): A[k].append(v)
    A={k:np.concatenate(v) for k,v in A.items()}; on=A['g']>thr
    rz=np.where(on,np.clip(np.clip(-0.1745*A['eh'],-0.3,0.3)-A['ab'],-0.2,0.2),A['st'])
    k=(np.abs(A['err'])>=0.5)&(np.abs(A['err'])<2)
    out=dict(err_best_epoch=ie['best_epoch'],gate_best_epoch=ig['best_epoch'],thr=round(thr,4),val_fpr=round(fpr_val,4),n_eval=int(len(rz)),
        toward=round(float(np.mean(np.sign(A['ab']+rz)==-np.sign(A['err']))),3),gate_on_frac_eval=round(float(on.mean()),3),
        sign_acc_0p5_2=round(float(np.mean(np.sign(A['eh'][k])==np.sign(A['err'][k]))),3),toward_current_residual=round(float(np.mean(np.sign(A['ab']+A['st'])==-np.sign(A['err']))),3))
    json.dump(out,open(f'{J}/train_check.json','w'),indent=1); print(json.dumps(out))
