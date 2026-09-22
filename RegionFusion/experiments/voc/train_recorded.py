import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from common import start_run,json_write,record_failure
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--iters',type=int,default=30000);p.add_argument('--tag',default='regionfusion_train')
    p.add_argument('--smoke',action='store_true');p.add_argument('--teacher-dir')
    a=p.parse_args(); cfg={}
    out=start_run(vars(a)|{'idea_config':cfg,'batch':4,'crop':320,'lr':1e-4,'weight_decay':.01,'w_diver':.1,
        'architecture':'ExCEL plus frozen C-RADIO spatial residual, zero-initialized projection',
        'radio_weights_sha256':'23e0c117de49d4ce909150fe6658d470829e6639647c7a5b035ce82e0d5b763c',
        'extra_encoder_at_inference':True},a.tag)
    try:
        env=os.environ.copy(); env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=1',
             'scripts/train_voc.py','--data_folder',a.data,'--max_iters',str(a.iters),'--seed',str(a.seed),
             '--work_dir',str(out/'training'),'--log_tag',a.tag,'--num_workers','4','--spg','4',
             '--log_iters','1' if a.smoke else ('20' if a.iters<100 else '200'),'--eval_iters','999999' if a.smoke else (str(a.iters) if a.iters<2000 else '2000')]
        if a.teacher_dir:cmd+=['--teacher_dir',str(Path(a.teacher_dir).resolve())]
        json_write(out/'command.json',{'argv':cmd,'cwd':str(ROOT),'env_overrides':{k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}})
        start=time.perf_counter()
        with (out/'stdout.log').open('w',buffering=1) as log,(out/'gpu_samples.jsonl').open('a',buffering=1) as gpu:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            json_write(out/'process.json',{'pid':proc.pid,'started_at':time.time(),
                                          'session_id':os.getsid(proc.pid),'detached':True})
            while proc.poll() is None:
                try:
                    sample=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader'],text=True)
                    gpu.write(json.dumps({'time':time.time(),'gpu':sample.strip()})+'\n')
                except Exception as exc:gpu.write(json.dumps({'error':str(exc)})+'\n')
                time.sleep(30)
        if proc.returncode:raise RuntimeError(f'train exit code {proc.returncode}; see stdout.log')
        checkpoints=list(out.glob(f'training/**/model_iter_{a.iters}.pth'))
        if not a.smoke and len(checkpoints)!=1:raise RuntimeError('Expected one final checkpoint')
        json_write(out/'metrics.json',{'status':'completed','stage':'training','iterations':a.iters,'seed':a.seed,
            'wall_seconds_including_validation':time.perf_counter()-start,'checkpoint':str(checkpoints[0]) if checkpoints else None,
            'smoke_only':a.smoke,
            'gpu_samples':'gpu_samples.jsonl','note':'Dedicated PyTorch peak values in stdout.log; nvidia-smi includes other jobs.'})
        json_write(out/'status.json',{'status':'completed','finished_at':time.time()})
    except BaseException:record_failure(out);raise
if __name__=='__main__':main()
