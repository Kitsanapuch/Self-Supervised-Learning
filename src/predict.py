"""Predict six-channel raw UCI-format windows with a saved model."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from src.experiment import Encoder,Classifier,CLASSES
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--input',type=Path,required=True,help='Numpy array [N,6,128], raw units, same signal order as training');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
meta=json.loads((a.run/'metadata.json').read_text())
if meta.get('synthetic'):raise ValueError('Synthetic smoke model is not for real inference')
x=np.load(a.input,allow_pickle=False).astype('float32')
if x.ndim!=3 or x.shape[1:]!=(6,128) or not np.isfinite(x).all():raise ValueError('Expected finite array [N,6,128]')
x=(x-np.array(meta['mean'],dtype='float32'))/np.array(meta['std'],dtype='float32')
model=Classifier(Encoder());model.load_state_dict(torch.load(a.run/'best.pt',map_location='cpu',weights_only=True));model.eval()
rows=[]
with torch.no_grad():
    for start in range(0,len(x),128):
        probabilities=model(torch.tensor(x[start:start+128])).softmax(1).numpy()
        for row in probabilities:
            label=int(row.argmax());rows.append({'class_id':label,'label':CLASSES[label],'probabilities':row.tolist()})
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2))
print(f'Saved {len(rows)} predictions; probabilities are not calibrated confidence guarantees.')
