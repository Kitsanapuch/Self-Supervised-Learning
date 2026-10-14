"""Aggregate real runs only, keeping distinct experimental settings separate."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(); p.add_argument('runs',type=Path); p.add_argument('--output',type=Path,default=Path('summary.csv')); a=p.parse_args()
groups=defaultdict(dict)
for file in a.runs.rglob('results.json'):
    result=json.loads(file.read_text())
    if result['synthetic']: continue
    meta=json.loads((file.parent/'metadata.json').read_text()); c=meta['config']
    for condition,values in result['test'].items():
        key=(c['mode'],c['mask'] if c['mode']=='ssl' else 'none',c['fraction'],condition,c['epochs'],c['pretrain_epochs'],c['lr'],c['batch_size'],c['device'],meta['data_sha256'],tuple(meta['val_subjects']))
        if c['seed'] in groups[key]: raise ValueError(f'Duplicate seed in same setting: {file}')
        groups[key][c['seed']]=values['macro_f1']
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['mode','mask','fraction','missing','epochs','pretrain_epochs','lr','batch_size','device','data_sha256','val_subjects','seeds','macro_f1_mean','macro_f1_sample_std'])
    for key,seeds in sorted(groups.items()):
        values=list(seeds.values()); w.writerow([*key,len(values),np.mean(values),np.std(values,ddof=1) if len(values)>1 else ''])
print(a.output)
