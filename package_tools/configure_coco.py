"""Configure a full-list COCO run after checking that required files exist."""
from pathlib import Path
import argparse,json,time,hashlib,math
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['train','eval'],required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--teacher-dir',type=Path);p.add_argument('--iters',type=int,default=100000);p.add_argument('--train-hours',type=float,default=168);p.add_argument('--eval-hours',type=float,default=168);a=p.parse_args()
 if a.iters<=0 or any(not math.isfinite(x) or x<=0 for x in [a.train_hours,a.eval_hours]):raise RuntimeError('iters and watchdog hours must be positive')
 lists=ROOT/'RegionFusion/datasets/coco';m=json.loads((ROOT/'configs/coco_full/manifest.json').read_text());split={}
 for name in ['train.txt','val.txt','val_internal.txt']:
  q=lists/name
  if hashlib.sha256(q.read_bytes()).hexdigest()!=m['split_hashes'][name]:raise RuntimeError('Split checksum mismatch: '+name)
  split[name]=q.read_text().split()
 missing={};examples=[]
 def require(group,path):
  if not path.is_file():
   missing[group]=missing.get(group,0)+1
   if len(examples)<5:examples.append(str(path))
 for ident in split['val.txt']:
  require('validation_images',a.data/'JPEGImages/val'/(ident+'.jpg'))
  require('validation_masks',a.data/'SegmentationClass/val'/(ident+'.png'))
 if a.mode=='train':
  if a.teacher_dir is None or not a.teacher_dir.is_dir():raise RuntimeError('Training needs an existing teacher pseudo directory covering the full training list')
  for ident in split['train.txt']:
   require('training_images',a.data/'JPEGImages/train'/(ident+'.jpg'))
   require('teacher_masks',a.teacher_dir/(ident+'.png'))
 if missing:raise RuntimeError('Full dataset files are incomplete: '+json.dumps({'missing':missing,'examples':examples}))
 out=ROOT/'outputs/coco_runtime/runtime_config.json'
 if not out.resolve().is_relative_to(ROOT):raise RuntimeError('Refusing to modify paths outside this package')
 now=time.time();train=a.mode=='train';hours=a.train_hours+a.eval_hours if train else a.eval_hours
 d={'train_n':m['train_n'],'val_n':m['val_n'],'internal_val_n':m['internal_val_n'],'max_student_steps':a.iters,'batch_size':4,'drop_last':False,'train_stop_epoch':now+a.train_hours*3600 if train else 0,'eval_stop_epoch':now+hours*3600,'deadline_epoch':now+(hours+1)*3600,'teacher_dir':str(a.teacher_dir.resolve()) if a.teacher_dir else None}
 out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(d,indent=2)+'\n');print('Full-list COCO runtime configuration ready. No model execution.')
if __name__=='__main__':main()
