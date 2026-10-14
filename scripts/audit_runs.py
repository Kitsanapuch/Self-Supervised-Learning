"""Independent integrity checks on saved experiment artifacts."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from src.experiment import Encoder,Classifier,load_archive,prepare,loader,score
root=Path(__file__).resolve().parents[1]; runs=root/'runs/full'
paths=sorted(runs.glob('*/results.json'));assert len(paths)==48,len(paths)
raw=load_archive(root/'data/uci_har.zip');splits,split_meta=prepare(raw)
archive_hash=hashlib.sha256((root/'data/uci_har.zip').read_bytes()).hexdigest()
records={}
for path in paths:
    folder=path.parent;m=json.loads((folder/'metadata.json').read_text());r=json.loads(path.read_text());h=json.loads((folder/'history.json').read_text());c=m['config']
    assert not r['synthetic']
    assert c['epochs']==30 and c['pretrain_epochs']==30 and c['lr']==.001 and c['batch_size']==128
    assert m['data_sha256']==archive_hash
    for key in ['train_subjects','val_subjects','test_subjects','mean','std']:assert m[key]==split_meta[key]
    idx=np.array(m['labeled_indices']);assert len(idx)==len(set(idx))==m['labeled_count']
    y=splits['train'][1];counts=np.bincount(y,minlength=6);actual=np.bincount(y[idx],minlength=6)
    np.testing.assert_array_equal(actual,np.ceil(counts*c['fraction']).astype(int))
    pretrain=[step for step in h if step['stage']=='pretrain']
    expected_pretrain=30 if c['mode']=='ssl' and not c.get('encoder_checkpoint') else 0
    assert len(pretrain)==expected_pretrain
    for step in h:assert np.isfinite(step['loss'])
    vals=[step['val_macro_f1'] for step in h if step['stage']=='finetune'];assert len(vals)==30
    assert max(vals)==r['best_val_macro_f1']
    model=Classifier(Encoder());model.load_state_dict(torch.load(folder/'best.pt',weights_only=True,map_location='cpu'))
    replay_val=score(model,loader(*splits['val'],128),'cpu')
    assert abs(replay_val['macro_f1']-r['best_val_macro_f1'])<1e-12
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    method=c['mask'] if c['mode']=='ssl' else 'supervised';records[method,c['fraction'],c['seed']]=m
    for condition,res in r['test'].items():
        cm=np.array(res['confusion_matrix']);assert cm.shape==(6,6) and (cm>=0).all() and cm.sum()==2947
        np.testing.assert_array_equal(cm.sum(1),np.bincount(raw['test'][1],minlength=6))
        f1=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)
        np.testing.assert_allclose(f1,res['class_f1'],atol=1e-12)
        assert abs(f1.mean()-res['macro_f1'])<1e-12
        assert abs(np.diag(cm).sum()/cm.sum()-res['accuracy'])<1e-12
for f in [.01,.05,.1,1.]:
 for seed in [0,1,2]:
    target=records['supervised',f,seed]['labeled_indices']
    assert all(records[method,f,seed]['labeled_indices']==target for method in ['time','channel','both'])
# Reload a checkpoint and independently reproduce all three reported test evaluations.
folder=runs/'both_f0.1_s0';model=Classifier(Encoder());model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True));torch.set_num_threads(2)
saved=json.loads((folder/'results.json').read_text())
for condition in ['none','acc','gyro']:
    replay=score(model,loader(*splits['test'],128),'cpu',condition)
    assert replay==saved['test'][condition],condition
out={'status':'passed','completed_runs':48,'evaluation_records':144,'checks':['finite losses and weights','class-stratified counts','paired labeled subsets','train-only normalization','subject-disjoint partitions','data checksum','all saved metric arithmetic','validation-selected best checkpoint score','independent checkpoint reload and three-condition metric replay'],'checkpoint_replay':'both_f0.1_s0'}
(root/'results/verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
