"""COCO student training using the configured complete task splits."""
from pathlib import Path
import argparse,json,os,random,sys,time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from common import start_run,json_write,record_failure,fingerprint,confusion,metrics
from datasets.coco_fixed import FixedCoco
from model.model_excel import ExCEL_model
from model.losses import get_seg_loss,get_aff_loss
from engine.optimizer_engine import build_optimizer
from utils.PAR import PAR
from utils.camutils import cure_attr_map,cams_to_affinity_label,get_mask_by_radius
from utils.affutils import refine_cams_with_aff,refine_cams_with_bkg_weclip
from utils import imutils

def model(mode='train'):
    return ExCEL_model(clip_model='ExCEL_ViT-B/16',in_channels=768,num_classes=81,dataset_name='ms_coco',num_atrr_clusters=224,json_file='./attributes_text/descriptors_ms_coco_gpt4.0_cluster_a_photo_of4.json',img_size=320,mode=mode,device='cuda').cuda()
def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def internal_val(net,loader):
    net.eval();h=np.zeros((81,81),np.int64)
    with torch.no_grad():
        for _,x,y,_ in loader:
            pred=net(F.interpolate(x.cuda(),(320,320),mode='bilinear',align_corners=False))[0]
            pred=F.interpolate(pred,size=y.shape[-2:],mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy()
            h+=confusion(y[0].numpy(),pred)
    net.train();return metrics(h)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--teacher-dir');p.add_argument('--iters',type=int,default=100000);p.add_argument('--seed',type=int,default=0);p.add_argument('--smoke',action='store_true');p.add_argument('--tag',required=True);a=p.parse_args()
    lists=ROOT/'datasets/coco';policy=json.loads((ROOT.parent/'outputs/coco_runtime/runtime_config.json').read_text());manifest=json.loads((ROOT.parent/'configs/coco_full/manifest.json').read_text())
    for f in ['train.txt','val.txt']:assert fingerprint(lists/f)==manifest['split_hashes'][f]
    out=start_run(vars(a)|{'dataset_subset':'coco_full_v1','num_classes':81,'batch_size':4,'drop_last':False,'crop':320,'max_iters':a.iters,'training_budget_kind':'upper_bound','cam_caa_threshold':.88,'cure_start':30000,'student_prediction_relation_start':80000,'initialization':'fresh CLIP plus C-RADIO for fused student; no VOC or trained seed student weight reuse','method':'H1 offline gated teacher replaces segmentation target only' if a.teacher_dir else 'Dense-fusion student with online ExCEL pseudo labels','policy':policy},a.tag)
    try:
        torch.set_num_threads(4);seed_all(a.seed);net=model();params=net.get_param_groups()
        from types import SimpleNamespace
        opt_args=SimpleNamespace(optimizer='PolyWarmupAdamW',lr=1e-4,wt_decay=.01,betas=(.9,.999),warmup_iters=200,max_iters=a.iters,warmup_lr=1e-6,power=1.)
        opt=build_optimizer(opt_args,params);par=PAR(num_iter=20,dilations=[1,2,4,8,12,24]).cuda();train=FixedCoco(a.data,lists,'train',a.teacher_dir,True);val=FixedCoco(a.data,lists,'val_internal')
        assert len(train)==manifest['train_n'] and len(val)==manifest['internal_val_n']
        loader=DataLoader(train,batch_size=4,shuffle=True,num_workers=4,drop_last=False,prefetch_factor=4)
        vloader=DataLoader(val,batch_size=1,shuffle=False,num_workers=4,drop_last=False)
        json_write(out/'training_ids.json',train.names);json_write(out/'validation_ids.json',(lists/'val.txt').read_text().split())
        provenance={'subset':'coco_full_v1','train_n':len(train),'val_n':manifest['val_n'],'training_list_sha256':fingerprint(lists/'train.txt'),'validation_list_sha256':fingerprint(lists/'val.txt'),'initialization':'fresh_pretrained_encoders_no_experiment_resume','max_iters':a.iters,'teacher_directory':a.teacher_dir,'drop_last':False,'batch_size':4}
        json_write(out/'data_provenance.json',provenance);seen=set();examples=0;iterator=iter(loader);net.train();radius=get_mask_by_radius(h=20,w=20,radius=8)
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter();checkpoint=None;stop_reason='upper_bound_completed';stop_evaluation=None
        with (out/'train.jsonl').open('a',buffering=1) as log:
            for step in range(1,a.iters+1):
                try:names,x,tags,box,teacher=next(iterator)
                except StopIteration:iterator=iter(loader);names,x,tags,box,teacher=next(iterator)
                seen.update(names);examples+=len(names);x=x.cuda();tags=tags.cuda();teacher=teacher.cuda();denorm=imutils.denormalize_img2(x.clone())
                seg,features,raw,attn,relation=net(x)
                if step-1>=30000:raw=cure_attr_map(net,x,ex_feats=features)
                pseudos=[]
                for i in range(len(names)):
                    refined,keys=refine_cams_with_aff(raw[i],attn[:,i],tags[i],size=x.shape[2:],seg_attn=relation[i:i+1] if step-1>=30000 else None,caa_thre=.88)
                    pseudo,_=refine_cams_with_bkg_weclip(refined,denorm[i],keys,par,size=x.shape[2:]);pseudos.append(pseudo)
                pseudo=torch.cat(pseudos);seg=F.interpolate(seg,pseudo.shape[1:],mode='bilinear',align_corners=False)
                seg_loss=get_seg_loss(seg,teacher.long() if a.teacher_dir else pseudo.long(),ignore_index=255)
                affinity_target=seg.detach().argmax(1) if a.teacher_dir and step-1>=80000 else pseudo
                aff=cams_to_affinity_label(affinity_target,mask=radius);diver,_,_=get_aff_loss(relation,aff);loss=seg_loss+.1*diver
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {step}')
                opt.zero_grad();loss.backward();opt.step()
                deadline_hit=time.time()>=policy['train_stop_epoch']
                if step%200==0 or a.smoke:
                    row={'step':step,'seg_loss':float(seg_loss.detach()),'diversity_loss':float(diver.detach()),'lr':opt.param_groups[0]['lr'],'seconds':time.perf_counter()-start,'observed_training_examples':examples,'observed_unique_training_images':len(seen)};log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                if not a.smoke and (step%2000==0 or step==a.iters or deadline_hit):
                    checkpoint=out/f'model_iter_{step}.pth';temporary=Path(str(checkpoint)+'.pending');torch.save(net.state_dict(),temporary);temporary.replace(checkpoint)
                    audit_dir=out/'checkpoint_audits'/str(step);audit_dir.mkdir(parents=True)
                    observed=[n for n in train.names if n in seen]
                    json_write(audit_dir/'observed_training_ids.json',observed)
                    json_write(audit_dir/'audit.json',provenance|{'role':'regionfusion_coco_student','branch':ROOT.name,'actual_steps':step,'observed_training_examples':examples,'observed_unique_training_images':len(seen),'counts_source':'measured_from_consumed_batches','checkpoint':str(checkpoint),'checkpoint_sha256':fingerprint(checkpoint),'cure_phase_used':step>30000,'late_relation_phase_used':step>80000})
                if not a.smoke and step%10000==0 and not deadline_hit:
                    value=internal_val(net,vloader);row={'step':step,'internal_single_scale_no_crf':value};log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                if deadline_hit:
                    stop_reason='wall_clock_budget';break
        observed=[n for n in train.names if n in seen];json_write(out/'observed_training_ids.json',observed);provenance.update(observed_unique_training_images=len(seen),observed_training_examples=examples,actual_steps=step,counts_source="measured_from_consumed_batches");json_write(out/'data_provenance.json',provenance)
        if not a.smoke:assert len(seen)==len(train)
        result={'status':'completed','stage':'training','dataset_subset':'coco_full_v1','train_n':len(train),'val_n':manifest['val_n'],'num_classes':81,'iterations':step,'planned_max_iterations':a.iters,'termination_reason':stop_reason,'selected_evaluation':stop_evaluation,'active_training_wall_seconds':time.perf_counter()-start,'seed':a.seed,'smoke_only':a.smoke,'checkpoint':str(checkpoint) if checkpoint else None,'wall_seconds':time.perf_counter()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'observed_training_examples':examples,'observed_unique_training_images':len(seen)}
        json_write(out/'metrics.json',result);json_write(out/'status.json',{'status':'completed','finished_at':time.time()});print(json.dumps(result),flush=True)
    except BaseException:record_failure(out);raise
if __name__=='__main__':main()
