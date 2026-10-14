"""Run 48 configurations, reusing only compatible pretrained encoders."""
import argparse, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=30); p.add_argument('--pretrain-epochs',type=int,default=30); p.add_argument('--device',default='cpu'); p.add_argument('--runs',type=Path,default=Path('runs/full')); a=p.parse_args()
a.runs.mkdir(parents=True,exist_ok=True)
for seed in [0,1,2]:
    for method in ['supervised','time','channel','both']:
        source=None
        for fraction in [.01,.05,.1,1.]:
            out=a.runs/f'{method}_f{fraction}_s{seed}'
            if (out/'results.json').exists():
                if method!='supervised' and source is None: source=out/'encoder.pt'
                continue
            if out.exists(): raise RuntimeError(f'Incomplete run: {out}; inspect and move it before retrying')
            cmd=[sys.executable,'-m','src.experiment','--mode','supervised' if method=='supervised' else 'ssl','--mask','time' if method=='supervised' else method,'--fraction',str(fraction),'--seed',str(seed),'--epochs',str(a.epochs),'--pretrain-epochs',str(a.pretrain_epochs),'--device',a.device,'--output',str(out)]
            if source is not None: cmd+=['--encoder-checkpoint',str(source)]
            print('RUN',out,flush=True)
            with (a.runs/(out.name+'.log')).open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            if method!='supervised' and source is None: source=out/'encoder.pt'
subprocess.run([sys.executable,'-m','src.summarize',str(a.runs),'--output',str(a.runs/'summary.csv')],check=True)
