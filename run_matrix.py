"""Run 48 configurations, reusing only compatible pretrained encoders."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=30); p.add_argument('--pretrain-epochs',type=int,default=30); p.add_argument('--device',default='cpu'); p.add_argument('--runs',type=Path,default=Path('runs/full')); a=p.parse_args()
a.runs.mkdir(parents=True,exist_ok=True)
data_path=Path('data/uci_har.zip')
if not data_path.exists(): raise FileNotFoundError('Run python -m src.download_data first')
data_hash=hashlib.sha256(data_path.read_bytes()).hexdigest()
for seed in [0,1,2]:
    for method in ['supervised','time','channel','both']:
        source=None
        for fraction in [.01,.05,.1,1.]:
            out=a.runs/f'{method}_f{fraction}_s{seed}'
            if (out/'results.json').exists():
                if not (out/'best.pt').exists() or not (out/'history.json').exists() or not (out/'metadata.json').exists():
                    raise RuntimeError(f'Incomplete saved artifacts: {out}')
                result=json.loads((out/'results.json').read_text())
                meta=json.loads((out/'metadata.json').read_text())
                config=meta['config']
                expected={'mode':'supervised' if method=='supervised' else 'ssl','mask':'time' if method=='supervised' else method,'fraction':fraction,'seed':seed,'epochs':a.epochs,'pretrain_epochs':a.pretrain_epochs,'device':a.device,'lr':.001,'batch_size':128}
                if result.get('synthetic') or any(config.get(key)!=value for key,value in expected.items()) or meta.get('data_sha256')!=data_hash:
                    raise RuntimeError(f'Existing run has a different protocol: {out}; use a new --runs directory')
                if method!='supervised' and source is None:
                    source=out/'encoder.pt'
                    if not source.exists(): raise FileNotFoundError(f'Missing pretrained encoder: {source}')
                continue
            if out.exists(): raise RuntimeError(f'Incomplete run: {out}; inspect and move it before retrying')
            cmd=[sys.executable,'-m','src.experiment','--mode','supervised' if method=='supervised' else 'ssl','--mask','time' if method=='supervised' else method,'--fraction',str(fraction),'--seed',str(seed),'--epochs',str(a.epochs),'--pretrain-epochs',str(a.pretrain_epochs),'--device',a.device,'--output',str(out)]
            if source is not None: cmd+=['--encoder-checkpoint',str(source)]
            print('RUN',out,flush=True)
            with (a.runs/(out.name+'.log')).open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            if method!='supervised' and source is None: source=out/'encoder.pt'
subprocess.run([sys.executable,'-m','src.summarize',str(a.runs),'--output',str(a.runs/'summary.csv')],check=True)
