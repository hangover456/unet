"""Run prepare -> short training -> full-volume inference -> evaluation."""
import json,time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from common import SPLIT,execute,parser,read_json,write_json,load_pair,plot_history,sha
from model import UNet2D

def resize(x,mode='bilinear',size=(256,256)):
    return F.interpolate(x,size=size,mode=mode,**({'align_corners':False} if mode=='bilinear' else {}))
def normalize(x):return (np.clip(x,-100,240).astype(np.float32)-70)/170
def prepare(a):
    cache=a.work/'unet_cache';cache.mkdir(parents=True,exist_ok=True)
    for case in SPLIT['train']+SPLIT['val']:
        dest=cache/f'{case}.npz'
        if dest.exists():continue
        _,x,y=load_pair(a.data,case);rng=np.random.default_rng(42+int(case))
        fg=np.flatnonzero((y>0).any((0,1)));bg=np.flatnonzero(~(y>0).any((0,1)))
        # Label-guided slice selection is training/validation sampling only.
        take=lambda v,n:rng.choice(v,min(n,len(v)),replace=False) if len(v) else np.array([],int)
        zs=np.sort(np.r_[take(fg,32),take(bg,16)])
        tx=resize(torch.from_numpy(normalize(x[:,:,zs]).transpose(2,0,1).copy())[:,None]).numpy().astype(np.float16)
        ty=resize(torch.from_numpy(y[:,:,zs].transpose(2,0,1).astype(np.float32).copy())[:,None],'nearest')[:,0].numpy().astype(np.uint8)
        np.savez_compressed(dest,x=tx,y=ty,slice_indices=zs);print('U-Net cached',case,len(zs),flush=True)
    write_json(cache/'contract.json',{'split':SPLIT,'source':str(a.data),'size':[256,256],'window':[-100,240],'labels':'unchanged 0/1/2','sample_per_case':'up to 32 foreground + 16 background slices'})

def loss(logits,y):
    probs=logits.float().softmax(1);gt=F.one_hot(y,3).permute(0,3,1,2).float()
    dice=(2*(probs*gt).sum((0,2,3))+1e-5)/(probs.sum((0,2,3))+gt.sum((0,2,3))+1e-5)
    return F.cross_entropy(logits.float(),y)+(1-dice[1:].mean())

def train(a):
    cache=a.work/'unet_cache';contract=read_json(cache/'contract.json')
    if contract['source']!=str(a.data) or contract['split']!=SPLIT:raise ValueError('Wrong preprocessing contract')
    def data(group):
        chunks=[np.load(cache/f'{c}.npz') for c in SPLIT[group]]
        return np.concatenate([p['x'] for p in chunks]),np.concatenate([p['y'] for p in chunks])
    tr,va=data('train'),data('val');rng=np.random.default_rng(42)
    def batch(data):
        x,y=data
        # Foreground oversampling changes training distribution, not test evaluation.
        pos=np.flatnonzero((y>0).any((1,2)))
        ids=[int(rng.choice(pos)) if len(pos) and rng.random()<.67 else int(rng.integers(len(y))) for _ in range(2)]
        return torch.from_numpy(x[ids].astype(np.float32)).cuda(),torch.from_numpy(y[ids].astype(np.int64)).cuda()
    net=UNet2D().cuda();opt=torch.optim.Adam(net.parameters(),lr=1e-3);scaler=torch.amp.GradScaler('cuda')
    history=[];best=-1.;updates=0;start=time.perf_counter()
    try:
        for epoch in range(a.max_epochs):
            net.train();ls=[]
            for _ in range(a.steps):
                x,y=batch(tr)
                if rng.random()<.5:x,y=x.flip(-1),y.flip(-1)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda'):v=loss(net(x),y)
                if not torch.isfinite(v):raise RuntimeError('Non-finite training loss')
                scaler.scale(v).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(net.parameters(),12)
                scaler.step(opt);scaler.update();updates+=1;ls.append(v.item())
                if time.perf_counter()-start>=a.minutes*60:break
            net.eval();vl=[];ds=[]
            with torch.no_grad():
                for _ in range(a.val_steps):
                    x,y=batch(va)
                    with torch.autocast('cuda'):out=net(x);v=loss(out,y)
                    pred=out.argmax(1);scores=[]
                    for k in (1,2):
                        gt,pr=y==k,pred==k;den=(gt.sum()+pr.sum()).item()
                        if den:scores.append(float((2*(gt&pr).sum()).item()/den))
                    vl.append(v.item());ds.append(float(np.mean(scores)) if scores else 1.)
            row=dict(epoch=epoch+1,train_loss=float(np.mean(ls)),val_loss=float(np.mean(vl)),patch_dice=float(np.mean(ds)),seconds=time.perf_counter()-start)
            history.append(row);write_json(a.run_dir/'history.json',history);print(row,flush=True)
            ck={'state_dict':net.state_dict(),'epoch':epoch+1,'classes':3,'base_width':32,'input_channels':1,'window':[-100,240],'input_size':256,'split':SPLIT}
            torch.save(ck,a.run_dir/'checkpoint_latest.pth')
            if row['patch_dice']>best:best=row['patch_dice'];torch.save(ck,a.run_dir/'checkpoint_best.pth')
            if time.perf_counter()-start>=a.minutes*60:break
        torch.save(ck,a.run_dir/'checkpoint_final.pth')
    except KeyboardInterrupt:
        if history:torch.save(ck,a.run_dir/'checkpoint_interrupted.pth')
        raise
    plot_history(a.run_dir)
    write_json(a.run_dir/'training_summary.json',{'status':'complete','epochs':len(history),'training_iterations':updates,'training_seconds':time.perf_counter()-start,
        'stop_reason':'time_budget' if time.perf_counter()-start>=a.minutes*60 else 'max_epochs','best_sampled_patch_dice':best,
        'checkpoint_final_sha256':sha(a.run_dir/'checkpoint_final.pth'),
        'parameter_count':sum(p.numel() for p in net.parameters()),'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
        'optimizer':'Adam','initial_lr':1e-3,'loss':'cross entropy + foreground soft Dice',
        'model':'PyTorch U-Net 2D, width32, 3 classes; adapted from legacy architecture, not weight compatible'})

def predict(a):
    net=UNet2D().cuda();ck=torch.load(a.run_dir/'checkpoint_final.pth',map_location='cpu',weights_only=False);net.load_state_dict(ck['state_dict']);net.eval()
    out=a.run_dir/'predictions';out.mkdir(exist_ok=True);start=time.perf_counter()
    with torch.no_grad():
        for case in SPLIT['val']+SPLIT['test']:
            # Inference reads only CT; labels are accessed later by evaluate().
            im=nib.load(a.data/'imagesTr'/f'pancreas_{case}.nii.gz');x=np.asarray(im.dataobj,dtype=np.float32);pred=np.empty(x.shape,dtype=np.uint8)
            for z in range(0,x.shape[2],4):
                batch=torch.from_numpy(normalize(x[:,:,z:z+4]).transpose(2,0,1).copy())[:,None].cuda()
                with torch.autocast('cuda'):logits=net(resize(batch))
                labels=resize(logits.float(),size=x.shape[:2]).argmax(1).cpu().numpy().astype(np.uint8)
                pred[:,:,z:z+4]=labels.transpose(1,2,0)
            h=im.header.copy();h.set_data_dtype(np.uint8);nib.save(nib.Nifti1Image(pred,im.affine,h),out/f'pancreas_{case}.nii.gz');print('Predicted full volume',case,flush=True)
    write_json(a.run_dir/'inference_summary.json',{'seconds':time.perf_counter()-start,'cases':SPLIT['val']+SPLIT['test'],'checkpoint':'checkpoint_final.pth'})

if __name__=='__main__':execute(parser().parse_args(),'unet',prepare,train,predict)
