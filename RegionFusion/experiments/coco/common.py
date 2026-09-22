from pathlib import Path
import datetime, hashlib, json, os, platform, subprocess, sys, time, traceback
import numpy as np
import torch

def json_write(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def fingerprint(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def start_run(config, tag):
    root=Path(__file__).resolve().parents[2]
    run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+tag
    out=root.parent/'outputs'/run_id
    out.mkdir(parents=True)
    for sub in ['visualizations','predictions']: (out/sub).mkdir()
    json_write(out/'config.json',config)
    json_write(out/'environment.json',dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
    (out/'code.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=root))
    manifest={str(p.relative_to(root)):fingerprint(p) for p in root.rglob('*.py') if '.git' not in p.parts}
    json_write(out/'code_sha256.json',manifest)
    (out/'environment.lock.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    try: (out/'gpu_start.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
    except Exception: pass
    json_write(out/'status.json',dict(status='running',started_at=time.time()))
    print('RUN_DIR='+str(out),flush=True)
    return out

def confusion(gt,pred):
    valid=(gt>=0)&(gt<81)
    assert gt.shape==pred.shape and np.isin(pred,np.arange(81)).all()
    return np.bincount(81*gt[valid].astype(np.int64)+pred[valid],minlength=6561).reshape(81,81)


def metrics(hist):
    tp=np.diag(hist).astype(float); gt=hist.sum(1); pr=hist.sum(0); union=gt+pr-tp
    def divide(a,b): return np.divide(a,b,out=np.full_like(a,np.nan,dtype=float),where=b>0)
    iou=divide(tp,union); precision=divide(tp,pr); recall=divide(tp,gt)
    clean=lambda a:[None if not np.isfinite(x) else float(x) for x in a]
    return dict(miou_percent=float(np.nanmean(iou[gt>0])*100),iou_percent=clean(iou*100),
        precision_percent=clean(precision*100),recall_percent=clean(recall*100),
        fg_false_discovery_percent=clean(divide(pr-tp,pr)[1:]*100),
        pixel_count=int(hist.sum()),present_classes=int((gt>0).sum()))

def record_failure(out):
    (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    json_write(out/'status.json',dict(status='failed',finished_at=time.time()))
