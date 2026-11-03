"""Shared data contract and report evidence for the two Task07 demonstrations."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, random, subprocess, sys, time, zipfile
from pathlib import Path
import nibabel as nib
import numpy as np

SPLIT = {'train': ['098','025','229','210','194','114','092','419'],
         'val': ['081','331'], 'test': ['035','032']}
LABELS = {'background': 0, 'pancreas': 1, 'tumour': 2}

def write_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def source_root(value):
    p=Path(value).expanduser().resolve()
    if not (p/'dataset.json').exists() and (p/'Task07_Pancreas/dataset.json').exists():p=p/'Task07_Pancreas'
    if not (p/'imagesTr').is_dir() or not (p/'labelsTr').is_dir():raise FileNotFoundError(f'Expected imagesTr and labelsTr in {p}')
    return p

def pair(root,case):return root/'imagesTr'/f'pancreas_{case}.nii.gz',root/'labelsTr'/f'pancreas_{case}.nii.gz'
def load_pair(root,case):
    a,b=pair(root,case);im,seg=nib.load(a),nib.load(b)
    if im.shape!=seg.shape or len(im.shape)!=3 or not np.allclose(im.affine,seg.affine,atol=1e-4):raise ValueError(f'Geometry mismatch: {case}')
    x=np.asarray(im.dataobj,dtype=np.float32);y=np.asarray(seg.dataobj)
    if not np.isfinite(x).all() or not set(np.unique(y))<={0,1,2}:raise ValueError(f'Invalid CT/labels: {case}')
    return im,x,y.astype(np.uint8)

def validate_split(split):
    if set(split)!={'train','val','test'}:raise ValueError('Need train/val/test')
    groups=[set(split[k]) for k in ('train','val','test')]
    if any(not v or len(v)!=len(split[k]) for k,v in zip(('train','val','test'),groups)):raise ValueError('Empty or duplicate split')
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):raise ValueError('Patient leakage')

def configure(args,model):
    validate_split(SPLIT)
    if args.minutes<=0 or args.steps<1 or args.val_steps<1 or args.max_epochs<1:raise ValueError('Budgets must be positive')
    args.data=source_root(args.data);args.work=Path(args.work).resolve();args.model=model
    os.environ.update(nnUNet_compile='false',nnUNet_n_proc_DA='0',nnUNet_def_n_proc='1',
                      nnUNet_wandb_enabled='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    import torch
    torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if not torch.cuda.is_available():raise RuntimeError(f'CUDA unavailable: {sys.executable}. Use the documented GPU interpreter.')
    torch.cuda.manual_seed_all(42)
    # New output per run; no automatic overwrite of earlier results.
    if args.run_dir:args.run_dir=Path(args.run_dir).resolve()
    elif args.stage in ('all','train'):
        args.run_dir=args.work/model/time.strftime('%Y%m%d_%H%M%S')
    elif args.stage in ('predict','collect'):
        args.run_dir=Path(read_json(args.work/model/'latest.json')['run_dir'])
    else:args.run_dir=None
    if args.run_dir:
        args.run_dir.mkdir(parents=True,exist_ok=True)
        if args.stage in ('all','train'):
            if (args.run_dir/'run.json').exists():raise FileExistsError('Run exists. Choose a fresh --run-dir.')
            write_json(args.work/model/'latest.json',{'run_dir':str(args.run_dir)})
            cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
            cfg.update(split=SPLIT,labels=LABELS,purpose='small-sample end-to-end execution demonstration; not a converged benchmark')
            write_json(args.run_dir/'config.json',cfg)
            try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parents[1],text=True).strip()
            except Exception:commit=None
            write_json(args.run_dir/'environment.json',{'python':sys.version,'interpreter':sys.executable,'torch':torch.__version__,
                'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
                'platform':platform.platform(),'git_commit':commit,'source_code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}})
    return args

def parser():
    p=argparse.ArgumentParser(description='Task07 small-sample three-class demonstration')
    p.add_argument('--stage',choices=['all','prepare','train','predict','collect'],default='all')
    p.add_argument('--data',default=str(Path.home()/'Desktop/data/Task07_Pancreas'))
    p.add_argument('--work',default=str(Path.home()/'Desktop/data/Task07_DemoRuns'))
    p.add_argument('--run-dir');p.add_argument('--minutes',type=float,default=90)
    p.add_argument('--steps',type=int,default=100);p.add_argument('--val-steps',type=int,default=10)
    p.add_argument('--max-epochs',type=int,default=1000)
    return p

def audit(args):
    dest=args.work/'shared';dest.mkdir(parents=True,exist_ok=True)
    result=dest/'dataset_audit.json'
    if result.exists():
        old=read_json(result)
        if old['source']!=str(args.data) or old['split']!=SPLIT:raise ValueError('Shared data contract changed; use a new --work')
        for c in old['cases']:
            for field,folder in [('image','imagesTr'),('label','labelsTr')]:
                p=args.data/folder/f"pancreas_{c['case']}.nii.gz"
                if sha(p)!=c[field+'_sha256']:raise ValueError(f'Source changed: {p}')
        return old
    rows=[];seen={}
    for group,cases in SPLIT.items():
        for case in cases:
            print('Checking',group,case,flush=True);im,x,y=load_pair(args.data,case);a,b=pair(args.data,case)
            digest=sha(a)
            if digest in seen:raise ValueError(f'Duplicate CT contents: {case}, {seen[digest]}')
            seen[digest]=case
            rows.append({'case':case,'split':group,'shape':list(im.shape),'spacing':[float(v) for v in im.header.get_zooms()],
                         'label_voxels':{str(k):int((y==k).sum()) for k in [0,1,2]},'image_sha256':digest,'label_sha256':sha(b)})
    report={'source':str(args.data),'split':SPLIT,'labels':LABELS,'cases':rows,'test_use':'only final prediction/evaluation; excluded from preprocessing statistics and training'}
    write_json(result,report);write_json(dest/'split.json',SPLIT);return report

def binary_metrics(gt,pred):
    tp=int(np.count_nonzero(gt&pred));fp=int(np.count_nonzero(~gt&pred));fn=int(np.count_nonzero(gt&~pred))
    return {'tp':tp,'fp':fp,'fn':fn,'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'gt_voxels':int(gt.sum()),'pred_voxels':int(pred.sum())}

def evaluate(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    rows=[]
    for group in ('val','test'):
        for case in SPLIT[group]:
            im,x,gt=load_pair(args.data,case);path=args.run_dir/'predictions'/f'pancreas_{case}.nii.gz';ni=nib.load(path)
            pred=np.asarray(ni.dataobj)
            if ni.shape!=im.shape or not np.allclose(ni.affine,im.affine,atol=1e-4) or not set(np.unique(pred))<={0,1,2}:raise ValueError(f'Prediction geometry/labels invalid: {case}')
            for label,name in [(1,'pancreas_label1'),(2,'tumour_label2'),(-1,'whole_pancreas_union')]:
                a,b=(gt>0,pred>0) if label==-1 else (gt==label,pred==label)
                rows.append(dict(split=group,case=case,target=name,**binary_metrics(a,b)))
            # Display selection uses GT for illustration only, never inference.
            areas=(gt>0).sum((0,1));z=int(areas.argmax());fig,ax=plt.subplots(1,3,figsize=(12,4),layout='constrained')
            for i,title in enumerate(['CT','Ground truth','Prediction']):
                ax[i].imshow(x[:,:,z].T,cmap='gray',vmin=-100,vmax=240,origin='lower')
                if i:
                    mask=gt[:,:,z].T if i==1 else pred[:,:,z].T
                    ax[i].imshow(np.ma.masked_where(mask==0,mask),cmap=ListedColormap(['black','lime','red']),vmin=0,vmax=2,alpha=.5,origin='lower')
                ax[i].set_title(title);ax[i].axis('off')
            fig.suptitle(f'{args.model} | {group} | {case} | z={z} selected by largest GT organ area\ngreen=pancreas, red=tumour; native array view')
            fig.savefig(args.run_dir/f'overlay_{group}_{case}.png',dpi=130);plt.close(fig)
    with (args.run_dir/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    means={g:{t:float(np.mean([r['dice'] for r in rows if r['split']==g and r['target']==t])) for t in ['pancreas_label1','tumour_label2','whole_pancreas_union']} for g in ['val','test']}
    write_json(args.run_dir/'metrics.json',{'per_case':rows,'mean_patient_dice':means,'scope':'full native volumes, no GT ROI crop, no test-time tuning, no postprocessing',
        'empty_convention':'both empty Dice/IoU=1; absent precision/recall denominator=null','purpose':'execution demonstration only'})
    print('Whole-volume Dice:',means,flush=True)

def plot_history(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h=read_json(run/'history.json');fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for k in ('train_loss','val_loss'):ax[0].plot([v[k] for v in h],label=k,marker='o',markersize=3)
    ax[0].legend();ax[0].set(xlabel='Epoch',ylabel='Loss')
    ax[1].plot([v['seconds'] for v in h],[v['patch_dice'] for v in h],marker='o',markersize=3);ax[1].set(xlabel='Training elapsed seconds',ylabel='Sampled validation patch Dice')
    fig.savefig(run/'learning_curves.png',dpi=140);plt.close(fig)

def collect(args):
    package=args.run_dir/'report_evidence.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for p in args.run_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.json','.csv','.png','.log','.txt','.md') and '.git' not in p.parts:
                z.write(p,p.relative_to(args.run_dir))
        for p in (args.work/'shared').glob('*.json'):z.write(p,'shared/'+p.name)
    print('REPORT PACKAGE:',package,flush=True)

class Tee:
    def __init__(self,stream,path):self.stream=stream;self.file=open(path,'a',encoding='utf-8',buffering=1)
    def write(self,s):self.stream.write(s);self.file.write(s)
    def flush(self):self.stream.flush();self.file.flush()

def execute(args,model,prepare,train,predict):
    args=configure(args,model)
    if args.run_dir:
        sys.stdout=Tee(sys.stdout,args.run_dir/'console.log');sys.stderr=Tee(sys.stderr,args.run_dir/'console.log')
    started=time.perf_counter()
    try:
        if args.stage in ('all','prepare','train','predict'):audit(args)
        if args.stage in ('all','prepare'):prepare(args)
        if args.stage in ('predict','collect'):
            saved=read_json(args.run_dir/'config.json')
            if saved['data']!=str(args.data) or saved['split']!=SPLIT:raise ValueError('Run data/split mismatch. Supply the original --data and --work.')
        if args.stage in ('all','train'):train(args)
        if args.stage in ('all','predict'):predict(args);evaluate(args)
        if args.run_dir:
            if args.stage!='collect':
                write_json(args.run_dir/'run.json',{'status':'complete','stage':args.stage,'total_stage_seconds':time.perf_counter()-started,'run_dir':str(args.run_dir)})
            collect(args)
    except BaseException as exc:
        if args.run_dir:
            import traceback
            write_json(args.run_dir/'run.json',{'status':'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed','error':str(exc),'traceback':traceback.format_exc()})
            collect(args)
        raise
