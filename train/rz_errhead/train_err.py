"""errsim_20261118 train: rz error head for the simulation arm (prereg.json). One ResMLP trunk (same inputs as residual_conv.pt:
hand-frame xy slots, wrench history blinded) with two outputs: e_hat = 15*tanh (signed yaw error to the nearer 180 deg, deg) and a gate
logit g (target = privileged teacher rotation active, i.e. any nonzero teacher rotation label). Loss = MSE(e_hat) on gate-positive rows with
|err| <= 15 + BCE(g) on all rows (sample_w weighted). Training sources and group split as train_nowr.py. CPU or GPU."""
import glob, json, os, sys, time, numpy as np, torch, torch.nn as nn
M='/home/dyros-recruit/mcy_ws/manifeel/manifeel'; sys.path.insert(0,M); O=f'{M}/data/outputs'; J=f'{O}/errsim_20261118'
from delta_residual_handframe_20261106 import to_hand
torch.manual_seed(0); dev='cuda:0' if torch.cuda.is_available() else 'cpu'; torch.set_num_threads(12)
yaw=lambda q: np.degrees(np.arctan2(2*(q[...,3]*q[...,2]+q[...,0]*q[...,1]),1-2*(q[...,1]**2+q[...,2]**2)))
w=lambda a:(a+180)%360-180; fo=lambda a: w(a-180*np.round(w(a)/180))
def rows(f):
    z=np.load(f); r=np.load(os.path.dirname(f)+'/rollout_traj.npz'); T=r['plug_quat'].shape[0]; ti=np.clip(z['t']-1,0,T-1); e=z['e']
    priv=fo(yaw(r['plug_quat'])-yaw(r['socket_quat']))[ti,e]
    ld,_=to_hand(z['ld'].astype(np.float32),z['delta_teacher'].astype(np.float32))
    X=np.concatenate([ld,z['img'].astype(np.float32),z['ff'].astype(np.float32)],1).astype(np.float16)
    gate=np.abs(z['delta_teacher'][:,3:6]).sum(1)>0
    return dict(X=X,priv=priv,gate=gate,sw=z['sample_w'].astype(np.float32),zone=z['zone'].astype(bool),ab=z['ab'][:,5],te=z['delta_teacher'][:,5],e=e)
class ErrNet(nn.Module):
    def __init__(s,d,mu,sd,h=256):
        super().__init__(); s.register_buffer('mu',torch.as_tensor(mu)); s.register_buffer('sd',torch.as_tensor(sd))
        s.net=nn.Sequential(nn.Linear(d,h),nn.LayerNorm(h),nn.ReLU(),nn.Linear(h,h),nn.LayerNorm(h),nn.ReLU(),nn.Linear(h,h),nn.LayerNorm(h),nn.ReLU(),nn.Linear(h,2))
    def forward(s,x):
        o=s.net((x-s.mu)/s.sd); return 15*torch.tanh(o[:,0]),o[:,1]
if __name__=='__main__':
    files=sorted(glob.glob(f'{O}/usb_keyed_s12r_gatefree_res_20261106/runs/COL_c*/delta_log.npz'))
    for S in ('usb_keyed_s13a_vgps_f15_20261107','usb_keyed_s13d_mcq_more_20261107','usb_keyed_s13e_mcq_t600_20261107'):
        files+=[f for f in sorted(glob.glob(f'{O}/{S}/runs/*_c*/delta_log.npz')) if os.path.exists(os.path.join(os.path.dirname(f),'eval_log.json'))]
    MAXF=int(os.environ.get('MAXF','0'))
    if MAXF: files=files[:MAXF]+files[-MAXF:]
    TR={k:[] for k in ('X','priv','gate','sw','G')}
    for gi,f in enumerate(files):
        d=rows(f)
        for k in ('X','priv','gate','sw'): TR[k].append(d[k])
        TR['G'].append(gi*1000+d['e'])
    TR={k:np.concatenate(v) for k,v in TR.items()}; N=len(TR['G']); print('rows',N,'gate+',int(TR['gate'].sum()),flush=True)
    ug=np.unique(TR['G']); rng=np.random.default_rng(0); rng.shuffle(ug); VA=np.isin(TR['G'],list(set(ug[:int(len(ug)*0.2)])))
    mu=TR['X'][~VA][::5].astype(np.float32).mean(0); sd=TR['X'][~VA][::5].astype(np.float32).std(0)+1e-3; B=list(range(21,93))+[99]; mu[B]=0; sd[B]=1e9
    net=ErrNet(TR['X'].shape[1],mu.astype(np.float32),sd.astype(np.float32)).to(dev); opt=torch.optim.Adam(net.parameters(),3e-4,weight_decay=1e-4)
    em=(TR['gate']&(np.abs(TR['priv'])<=15)).astype(np.float32); y=np.clip(TR['priv'],-15,15).astype(np.float32); bce=nn.functional.binary_cross_entropy_with_logits
    def run(idx,train):
        net.train(train); tot=te=tg=0.0; ws=0.0
        for b in range(0,len(idx),256 if train else 8192):
            k=idx[b:b+(256 if train else 8192)]; x=torch.from_numpy(TR['X'][k].astype(np.float32)).to(dev)
            yy,mm,gg,sw=(torch.from_numpy(a[k]).to(dev) for a in (y,em,TR['gate'].astype(np.float32),TR['sw']))
            with torch.set_grad_enabled(train):
                eh,g=net(x); le=((eh-yy)**2*mm).sum()/mm.sum().clamp(min=1); lg=(bce(g,gg,reduction='none')*sw).sum()/sw.sum(); l=le/10+lg
                if train: opt.zero_grad(); l.backward(); opt.step()
            tot+=float(l)*len(k); te+=float(le)*len(k); tg+=float(lg)*len(k); ws+=len(k)
        return tot/ws,te/ws,tg/ws
    tri,vai=np.flatnonzero(~VA),np.flatnonzero(VA); best=(1e9,None,-1); bad=0; hist=[]
    for ep in range(int(os.environ.get('MAXEP','20'))):
        t0=time.time(); lt=run(np.random.default_rng(ep).permutation(tri),True); lv=run(vai,False); hist.append([ep,*[round(v,4) for v in lt+lv],round(time.time()-t0)]); print(hist[-1],flush=True)
        if lv[0]<best[0]: best=(lv[0],{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},ep); bad=0
        else:
            bad+=1
            if bad>=4: break
    torch.save(dict(model=best[1],d=int(TR['X'].shape[1]),best_epoch=best[2],hist=hist),f'{J}/errnet.pt'); del TR
    # offline gate (prereg E0): Q400B eval rows, zone, teacher rz != 0, |err| <= 5
    net.load_state_dict(best[1]); net.eval(); src=lambda s:f'{O}/evq_20261117/eval' if s==20260921 else f'{O}/b2_20261117/eval'
    A=dict(err=[],ab=[],eh=[],g=[],pos=[]); FP=[]
    for s in (20260921,20260922,20260923):
        for c in [76700+25*i for i in range(8)]:
            d=rows(f'{src(s)}/Q400B_s{s}_c{c}/delta_log.npz')
            with torch.no_grad(): eh,g=(t.cpu().numpy() for t in net(torch.from_numpy(d['X'].astype(np.float32)).to(dev)))
            m=d['zone']&(np.abs(d['te'])>1e-6)&(np.abs(d['priv'])<=5)
            for k,v in (('err',d['priv']),('ab',d['ab']),('eh',eh),('g',g)): A[k].append(v[m])
            FP.append(g[~d['gate']]>0)
    A={k:np.concatenate(v) for k,v in A.items() if v}; on=A['g']>0
    rz=np.where(on,np.clip(np.clip(-0.1745*A['eh'],-0.3,0.3)-A['ab'],-0.2,0.2),0.0)
    out=dict(best_epoch=best[2],n_eval=int(len(rz)),toward_gated=round(float(np.mean(np.sign(A['ab']+rz)==-np.sign(A['err']))),3),
        gate_on_frac_eval_rows=round(float(on.mean()),3),gate_on_frac_teacher_off_rows=round(float(np.concatenate(FP).mean()),3),
        sign_acc_0p5_2=round(float(np.mean(np.sign(A['eh'])==np.sign(A['err']))[None][0]),3))
    k=(np.abs(A['err'])>=0.5)&(np.abs(A['err'])<2); out['sign_acc_0p5_2']=round(float(np.mean(np.sign(A['eh'][k])==np.sign(A['err'][k]))),3)
    json.dump(out,open(f'{J}/train_check.json','w'),indent=1); print(json.dumps(out))
