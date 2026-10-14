"""Exploratory paired cluster bootstrap on the nine test subjects.
Conditional on fixed train/validation split and three fitted seeds; no multiplicity correction.
"""
import csv,json
from pathlib import Path
import numpy as np
import torch
from src.experiment import Encoder,Classifier,load_archive,prepare,loader
root=Path(__file__).resolve().parents[1];raw=load_archive(root/'data/uci_har.zip');splits,_=prepare(raw)
subjects=raw['test'][2];ids=np.unique(subjects);x,y=splits['test'];torch.set_num_threads(2)
methods=['supervised','time','channel','both'];fractions=[.01,.05,.1,1.];seeds=[0,1,2];cms={}
for m in methods:
 for f in fractions:
  matrices=[]
  for seed in seeds:
   folder=root/'runs/full'/f'{m}_f{f}_s{seed}'
   model=Classifier(Encoder());model.load_state_dict(torch.load(folder/'best.pt',weights_only=True,map_location='cpu'));model.eval();pred=[]
   with torch.no_grad():
    for xb,_ in loader(x,y,128):pred.extend(model(xb).argmax(1).numpy().tolist())
   pred=np.array(pred); per_subject=[]
   for subject in ids:
    cm=np.zeros((6,6),dtype=int);mask=subjects==subject;np.add.at(cm,(y[mask],pred[mask]),1);per_subject.append(cm)
   recorded=json.loads((folder/'results.json').read_text())['test']['none']['confusion_matrix']
   np.testing.assert_array_equal(np.sum(per_subject,axis=0),np.array(recorded))
   matrices.append(per_subject)
  cms[m,f]=np.array(matrices)
# Keep all within-subject windows and pair draws across methods and seeds.
rng=np.random.default_rng(20261014);draws=rng.integers(0,len(ids),size=(5000,len(ids)))
def f1(cm):
 tp=np.diagonal(cm,axis1=-2,axis2=-1)
 return (2*tp/np.maximum(cm.sum(-1)+cm.sum(-2),1)).mean(-1)
rows=[]
for f in fractions:
 base=cms['supervised',f]
 base_boot=f1(base[:,draws].sum(axis=2)).mean(axis=0)*100
 for m in methods[1:]:
  current=cms[m,f];observed=(f1(current.sum(axis=1))-f1(base.sum(axis=1))).mean()*100
  boot=f1(current[:,draws].sum(axis=2)).mean(axis=0)*100-base_boot
  low,high=np.percentile(boot,[2.5,97.5])
  rows.append({'fraction':f,'method':m,'delta_macro_f1_pp':observed,'cluster_ci_low_pp':low,'cluster_ci_high_pp':high,'resamples':5000,'subjects':len(ids)})
(root/'results').mkdir(exist_ok=True)
with (root/'results/subject_bootstrap.csv').open('w',newline='') as fp:
 w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
np.savez_compressed(root/'results/subject_confusions.npz',subject_ids=ids,**{f'{m}_f{f}':v for (m,f),v in cms.items()})
(root/'results/subject_bootstrap_notes.json').write_text(json.dumps({'status':'exploratory supplemental analysis','resampling_unit':'test subject, with replacement','test_subjects':ids.tolist(),'replicates':5000,'seed':20261014,'interval':'paired percentile 95%','scope':'conditional on fixed training split and fitted seeds; captures test-subject sampling variability only','limitations':['only nine subjects','not a population guarantee','no multiple-comparison correction','training-split uncertainty not captured']},indent=2))
print('Completed paired subject bootstrap for 12 clean-condition SSL comparisons.')
