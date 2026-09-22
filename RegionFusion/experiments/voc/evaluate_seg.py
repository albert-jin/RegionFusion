"""Final VOC val segmentation without image-label masking; official multiscale settings."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from common import start_run,json_write,confusion,metrics,record_failure,fingerprint
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--checkpoint',required=True)
    p.add_argument('--tag',default='seg_val'); p.add_argument('--limit',type=int,default=0); args=p.parse_args()
    out=start_run(vars(args)|{'scales':[0.7,1.0,1.2,1.5],'image_label_masking':False,'crf':True,
        'protocol':'official: scale1 uses unflipped logits; other scales average flips; 320x320 base'},args.tag)
    try:
        from datasets import voc
        from model.model_excel import ExCEL_model
        from utils.dcrf import DenseCRF
        from utils import imutils
        torch.set_num_threads(4); torch.manual_seed(0); os.chdir(ROOT)
        model=ExCEL_model(clip_model='ExCEL_ViT-B/16',in_channels=768,img_size=320,mode='val',device='cuda').cuda().eval()
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        state={k.removeprefix('module.'):v for k,v in state.items() if 'encoder.visual.positional_embedding' not in k}
        missing=model.load_state_dict(state,strict=False)
        if missing.unexpected_keys or any(k!='encoder.visual.positional_embedding' for k in missing.missing_keys): raise RuntimeError(str(missing))
        json_write(out/'checkpoint.json',{'path':args.checkpoint,'sha256':fingerprint(args.checkpoint),'load':str(missing)})
        ds=voc.VOC12SegDataset(root_dir=args.data,name_list_dir=str(ROOT/'datasets/voc'),split='val',stage='val',aug=False)
        crf=DenseCRF(iter_max=10,pos_xy_std=1,pos_w=3,bi_xy_std=67,bi_rgb_std=3,bi_w=4)
        rawh=[]; crfh=[]; names=[]; count=len(ds) if not args.limit else min(args.limit,len(ds))
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
        with torch.no_grad(),(out/'progress.jsonl').open('a',buffering=1) as f:
            for i in range(count):
                name,img,gt,_=ds[i]; names.append(name); x=torch.from_numpy(img).unsqueeze(0).cuda(); seg_list=[]
                for sc in [1.,.7,1.2,1.5]:
                    size=int(320*sc); xx=F.interpolate(x,size=(size,size),mode='bilinear',align_corners=False)
                    logits=model(torch.cat([xx,xx.flip(-1)],0))[0]
                    logits=F.interpolate(logits,size=gt.shape,mode='bilinear',align_corners=False)
                    seg=logits[:1] if sc==1. else (logits[:1]+logits[1:].flip(-1))/2
                    seg_list.append(seg)
                seg=torch.stack(seg_list).mean(0); pred=seg.argmax(1)[0].cpu().numpy().astype(np.uint8)
                # pydensecrf requires a writable, C-contiguous uint8 buffer.
                rgb=np.array(Image.open(Path(args.data)/'JPEGImages'/f'{name}.jpg').convert('RGB'),dtype=np.uint8,copy=True,order='C')
                refined=crf(rgb,seg.softmax(1)[0].cpu().numpy()).argmax(0).astype(np.uint8)
                rawh.append(confusion(gt,pred)); crfh.append(confusion(gt,refined))
                if i<8:
                    canvas=np.concatenate([rgb,imutils.encode_cmap(gt).astype(np.uint8),imutils.encode_cmap(pred).astype(np.uint8),imutils.encode_cmap(refined).astype(np.uint8)],1)
                    Image.fromarray(canvas).save(out/'visualizations'/f'{name}_image_gt_raw_crf.png')
                row={'n':i+1,'name':name,'raw_miou':metrics(np.sum(rawh,0))['miou_percent'],'crf_miou':metrics(np.sum(crfh,0))['miou_percent']}
                f.write(json.dumps(row)+'\n')
                if (i+1)%50==0: print(json.dumps(row),flush=True)
        torch.cuda.synchronize()
        result={'status':'completed','n':count,'split':'val','raw':metrics(np.sum(rawh,0)),
            'crf':metrics(np.sum(crfh,0)),'wall_seconds':time.perf_counter()-start,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'complete_split':not args.limit}
        json_write(out/'image_ids.json',names); np.savez_compressed(out/'confusions.npz',raw=rawh,crf=crfh)
        json_write(out/'metrics.json',result); json_write(out/'status.json',{'status':'completed','finished_at':time.time()})
        print(json.dumps(result),flush=True)
    except BaseException: record_failure(out); raise
if __name__=='__main__':main()
