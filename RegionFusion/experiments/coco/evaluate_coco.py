"""COCO full validation entry; no image-tag mask at inference."""
from pathlib import Path
import argparse,json,os,sys,time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from common import start_run,json_write,record_failure,fingerprint,confusion,metrics
from train_coco import model
from datasets.coco_fixed import FixedCoco
from utils.dcrf import DenseCRF
from utils import imutils

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--tag',required=True);a=p.parse_args()
    out=start_run(vars(a)|{'num_classes':81,'dataset_subset':'coco_full_v1','image_label_masking':False,'scales':[.7,1.,1.2,1.5],'crf':True,'validation_ids_sha256':fingerprint(ROOT/'datasets/coco/val.txt'),'protocol':'ExCEL released COCO: all four scales average horizontal flips; logits downsampled to 0.2 original H/W before averaging and upsampled for scoring/CRF'},a.tag)
    try:
        torch.set_num_threads(4);torch.manual_seed(0);net=model(mode='val').eval();weights=torch.load(a.checkpoint,map_location='cpu',weights_only=True);weights={k.removeprefix('module.'):v for k,v in weights.items() if 'encoder.visual.positional_embedding' not in k};msg=net.load_state_dict(weights,strict=False)
        assert not msg.unexpected_keys and all(k=='encoder.visual.positional_embedding' for k in msg.missing_keys);json_write(out/'checkpoint.json',{'path':a.checkpoint,'sha256':fingerprint(a.checkpoint),'load':str(msg)})
        policy=json.loads((ROOT.parent/'outputs/coco_runtime/runtime_config.json').read_text());ds=FixedCoco(a.data,ROOT/'datasets/coco','val');assert len(ds)==policy['val_n'];n=min(a.limit,len(ds)) if a.limit else len(ds)
        crf=DenseCRF(iter_max=10,pos_xy_std=1,pos_w=3,bi_xy_std=67,bi_rgb_std=3,bi_w=4);rawh=[];crfh=[];ids=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(),(out/'progress.jsonl').open('a',buffering=1) as log:
            for i in range(n):
                if time.time()>=policy['eval_stop_epoch']:raise RuntimeError('Deadline: incomplete evaluation is not a final result')
                name,x,gt,_=ds[i];ids.append(name);x=torch.from_numpy(x)[None].cuda();low=(max(1,int(gt.shape[0]*.2)),max(1,int(gt.shape[1]*.2)));logits=[]
                for scale in [1.,.7,1.2,1.5]:
                    xx=F.interpolate(x,(int(320*scale),int(320*scale)),mode='bilinear',align_corners=False);s=net(torch.cat([xx,xx.flip(-1)]))[0];s=F.interpolate(s,low,mode='bilinear',align_corners=False);logits.append((s[:1]+s[1:].flip(-1))/2)
                s=F.interpolate(torch.stack(logits).mean(0),gt.shape,mode='bilinear',align_corners=False);raw=s.argmax(1)[0].cpu().numpy().astype(np.uint8);rgb=np.array(Image.open(Path(a.data)/'JPEGImages/val'/(name+'.jpg')).convert('RGB'),copy=True,order='C');refined=crf(rgb,s.softmax(1)[0].cpu().numpy()).argmax(0).astype(np.uint8)
                rawh.append(confusion(gt,raw));crfh.append(confusion(gt,refined))
                if i<8:Image.fromarray(np.concatenate([rgb,imutils.encode_cmap(gt).astype(np.uint8),imutils.encode_cmap(raw).astype(np.uint8),imutils.encode_cmap(refined).astype(np.uint8)],1)).save(out/'visualizations'/(name+'.png'))
                row={'n':i+1,'name':name,'raw_miou':metrics(np.sum(rawh,0))['miou_percent'],'crf_miou':metrics(np.sum(crfh,0))['miou_percent']};log.write(json.dumps(row)+'\n')
                if (i+1)%100==0:print(json.dumps(row),flush=True)
        json_write(out/'image_ids.json',ids);np.savez_compressed(out/'confusions.npz',raw=rawh,crf=crfh)
        m={'status':'completed','stage':'final_student_segmentation','dataset_subset':'coco_full_v1','split':'val','n':n,'complete_split':not a.limit,'num_classes':81,'raw':metrics(np.sum(rawh,0)),'crf':metrics(np.sum(crfh,0)),'wall_seconds':time.perf_counter()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
        json_write(out/'metrics.json',m);json_write(out/'status.json',{'status':'completed','finished_at':time.time()});print(json.dumps(m),flush=True)
    except BaseException:record_failure(out);raise
if __name__=='__main__':main()
