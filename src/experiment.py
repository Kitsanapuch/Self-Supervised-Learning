"""Reproducible UCI HAR supervised / masked-reconstruction SSL experiments."""
import argparse, csv, hashlib, io, json, math, platform, subprocess, time, urllib.request, zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

URL = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
SIGNALS = [f'{kind}_{axis}' for kind in ('total_acc', 'body_gyro') for axis in 'xyz']
CLASSES = ['walking', 'walking_upstairs', 'walking_downstairs', 'sitting', 'standing', 'laying']

def download(folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'uci_har.zip'
    if not path.exists():
        temp = path.with_suffix('.tmp')
        urllib.request.urlretrieve(URL, temp)
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None:
                raise ValueError('Corrupt dataset archive')
        temp.replace(path)
    return path

def load_archive(path):
    # Read without extracting untrusted paths; support UCI's nested archive.
    with zipfile.ZipFile(path) as outer:
        nested = [n for n in outer.namelist() if n.endswith('UCI HAR Dataset.zip')]
        payload = io.BytesIO(outer.read(nested[0])) if nested else path
    result = {}
    with zipfile.ZipFile(payload) as z:
        def read(suffix):
            matches = [n for n in z.namelist() if n.endswith(suffix) and not n.startswith('__MACOSX')]
            if len(matches) != 1:
                raise ValueError(f'Expected one file: {suffix}, got {matches}')
            return np.loadtxt(io.BytesIO(z.read(matches[0])))
        for part in ('train', 'test'):
            x = np.stack([read(f'{part}/Inertial Signals/{s}_{part}.txt') for s in SIGNALS], axis=1).astype('float32')
            y = read(f'{part}/y_{part}.txt').astype('int64') - 1
            subject = read(f'{part}/subject_{part}.txt').astype('int64')
            result[part] = (x, y, subject)
    return result

def prepare(raw, split_seed=2026):
    x, y, people = raw['train']
    test_x, test_y, test_people = raw['test']
    ids = np.unique(people)
    val_ids = np.random.default_rng(split_seed).permutation(ids)[:max(1, round(len(ids)*.2))]
    v = np.isin(people, val_ids)
    assert not set(people).intersection(test_people)
    assert not set(people[v]).intersection(people[~v])
    mean = x[~v].mean(axis=(0, 2), keepdims=True)
    std = x[~v].std(axis=(0, 2), keepdims=True).clip(1e-6)
    splits = {'train': ((x[~v]-mean)/std, y[~v]), 'val': ((x[v]-mean)/std, y[v]), 'test': ((test_x-mean)/std, test_y)}
    metadata = {'train_subjects': np.unique(people[~v]).tolist(), 'val_subjects': val_ids.tolist(), 'test_subjects': np.unique(test_people).tolist(), 'mean': mean.tolist(), 'std': std.tolist()}
    return splits, metadata

def subset(y, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError('fraction must be in (0,1]')
    rng = np.random.default_rng(seed)
    selected = []
    for c in range(6):
        candidates = np.flatnonzero(y == c)
        if len(candidates) == 0:
            raise ValueError(f'Missing training class {c}')
        selected.extend(rng.permutation(candidates)[:max(1, math.ceil(fraction*len(candidates)))])
    return np.array(sorted(selected))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(6,32,7,padding=3), nn.GroupNorm(4,32), nn.GELU(), nn.Conv1d(32,64,5,padding=2), nn.GroupNorm(8,64), nn.GELU())
    def forward(self, x):
        return self.net(x)

class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder, self.head = encoder, nn.Linear(64,6)
    def forward(self,x):
        return self.head(self.encoder(x).mean(-1))

def mask_input(x, mode):
    mask = torch.zeros_like(x, dtype=torch.bool)
    b,c,t = x.shape
    if mode in ('time','both'):
        width = max(1, t//4)
        starts = torch.randint(t-width+1,(b,),device=x.device)
        positions = torch.arange(t,device=x.device)[None,:]
        mask |= ((positions >= starts[:,None]) & (positions < starts[:,None]+width))[:,None,:]
    if mode in ('channel','both'):
        channels = torch.randint(c,(b,),device=x.device)
        mask[torch.arange(b,device=x.device), channels, :] = True
    return x.masked_fill(mask,0), mask

def loader(x,y,batch,shuffle=False,seed=0):
    return DataLoader(TensorDataset(torch.tensor(x),torch.tensor(y)),batch_size=batch,shuffle=shuffle,generator=torch.Generator().manual_seed(seed))

def score(model, data, device, missing='none'):
    model.eval()
    confusion = np.zeros((6,6),dtype=np.int64)
    with torch.no_grad():
        for x,y in data:
            x = x.to(device)
            if missing == 'acc': x[:,:3] = 0
            if missing == 'gyro': x[:,3:] = 0
            prediction = model(x).argmax(1).cpu().numpy()
            np.add.at(confusion,(y.numpy(),prediction),1)
    tp = np.diag(confusion)
    f1 = 2*tp/np.maximum(confusion.sum(0)+confusion.sum(1),1)
    return {'macro_f1': float(f1.mean()), 'accuracy': float(tp.sum()/max(confusion.sum(),1)), 'class_f1': f1.tolist(), 'confusion_matrix': confusion.tolist()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',choices=['supervised','ssl'],default='supervised')
    p.add_argument('--mask',choices=['time','channel','both'],default='time')
    p.add_argument('--fraction',type=float,default=.1)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--pretrain-epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu')
    p.add_argument('--encoder-checkpoint',type=Path,help='Reuse encoder from an identical pretraining configuration')
    p.add_argument('--smoke',action='store_true',help='Synthetic integration check; not research evidence')
    args = p.parse_args()
    started = time.perf_counter()
    if args.epochs < 1 or args.pretrain_epochs < 1 or args.batch_size < 1:
        p.error('epochs and batch size must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.smoke:
        rng = np.random.default_rng(42)
        def fake(n):
            return (rng.normal(size=(n,6,128)).astype('float32'),np.arange(n)%6)
        splits = {k:fake(n) for k,n in [('train',60),('val',24),('test',24)]}
        meta = {'synthetic':True}; checksum = None
    else:
        path = download(args.data_dir)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        splits,meta = prepare(load_archive(path))
    idx = subset(splits['train'][1],args.fraction,args.seed)
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
        dirty = bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    except (OSError,subprocess.CalledProcessError): commit,dirty = None,None
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    meta.update({'dataset_provenance': json.loads((args.data_dir/'provenance.json').read_text()) if (args.data_dir/'provenance.json').exists() else {'download_url':URL},'config':config,'git_commit':commit,'git_dirty':dirty,'data_sha256':checksum,'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'labeled_indices':idx.tolist(),'labeled_count':len(idx),'actual_fraction':len(idx)/len(splits['train'][1]),'classes':CLASSES,'signals':SIGNALS})
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2))
    train_x,train_y = splits['train']
    train = loader(train_x[idx],train_y[idx],args.batch_size,True,args.seed)
    val = loader(*splits['val'],args.batch_size)
    encoder = Encoder().to(args.device)
    history = []
    if args.encoder_checkpoint:
        if args.mode != 'ssl': raise ValueError('Encoder checkpoint requires SSL mode')
        prior=json.loads((args.encoder_checkpoint.parent/'metadata.json').read_text())
        for key in ['seed','mask','pretrain_epochs','lr','batch_size','device','smoke']:
            if prior['config'][key] != config[key]: raise ValueError(f'Incompatible pretraining: {key}')
        if prior['data_sha256'] != checksum or prior.get('train_subjects') != meta.get('train_subjects'):
            raise ValueError('Pretraining dataset or subjects mismatch')
        encoder.load_state_dict(torch.load(args.encoder_checkpoint,map_location=args.device,weights_only=True))
        meta['encoder_source_sha256']=hashlib.sha256(args.encoder_checkpoint.read_bytes()).hexdigest()
        (args.output/'metadata.json').write_text(json.dumps(meta,indent=2))
    if args.mode == 'ssl' and not args.encoder_checkpoint:
        # No labels are passed to the pretraining objective.
        unlabelled = loader(train_x,np.zeros(len(train_x),dtype='int64'),args.batch_size,True,args.seed)
        decoder = nn.Conv1d(64,6,1).to(args.device)
        optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=args.lr)
        for epoch in range(args.pretrain_epochs):
            encoder.train(); decoder.train(); total,count = 0.,0
            for x,_ in unlabelled:
                x=x.to(args.device); corrupted, mask=mask_input(x,args.mask)
                loss=((decoder(encoder(corrupted))-x)[mask]**2).mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total+=loss.item()*len(x); count+=len(x)
            history.append({'stage':'pretrain','epoch':epoch+1,'loss':total/count})
        torch.save(encoder.state_dict(),args.output/'encoder.pt')
    # Same head initialization across methods; full encoder fine-tuning.
    torch.manual_seed(args.seed+1000)
    model=Classifier(encoder).to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    best=-1.
    for epoch in range(args.epochs):
        model.train(); total,count=0.,0
        for x,y in train:
            x,y=x.to(args.device),y.to(args.device)
            loss=nn.functional.cross_entropy(model(x),y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total+=loss.item()*len(x); count+=len(x)
        metrics=score(model,val,args.device)
        history.append({'stage':'finetune','epoch':epoch+1,'loss':total/count,'val_macro_f1':metrics['macro_f1']})
        if metrics['macro_f1']>best:
            best=metrics['macro_f1']; torch.save(model.state_dict(),args.output/'best.pt')
        print(f'epoch={epoch+1} val_macro_f1={metrics["macro_f1"]:.4f}',flush=True)
    model.load_state_dict(torch.load(args.output/'best.pt',map_location=args.device,weights_only=True))
    test=loader(*splits['test'],args.batch_size)
    results={'elapsed_seconds':time.perf_counter()-started,'synthetic':args.smoke,'best_val_macro_f1':best,'test':{condition:score(model,test,args.device,condition) for condition in ['none','acc','gyro']}}
    (args.output/'results.json').write_text(json.dumps(results,indent=2))
    (args.output/'history.json').write_text(json.dumps(history,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
