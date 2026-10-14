"""Fetch the exact benchmark archive used in the supplied experiment results."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
provenance=json.loads((ROOT/'docs/DATA_PROVENANCE.json').read_text())
folder=ROOT/'data';folder.mkdir(exist_ok=True);target=folder/'uci_har.zip'
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while chunk:=f.read(1024*1024):h.update(chunk)
 return h.hexdigest()
if target.exists():
 if digest(target)!=provenance['sha256']:
  raise ValueError('Existing archive differs from the recorded source. Move it before downloading; it will not be overwritten.')
else:
 temp=folder/'download.tmp'
 try:
  with urllib.request.urlopen(provenance['download_url'],timeout=60) as response,temp.open('wb') as output:
   while chunk:=response.read(1024*1024):output.write(chunk)
  if digest(temp)!=provenance['sha256']:raise ValueError('Archive checksum mismatch')
  temp.replace(target)
 finally:
  if temp.exists():temp.unlink()
(folder/'provenance.json').write_text(json.dumps(provenance,indent=2))
print('Dataset ready; SHA-256 verified:',provenance['sha256'])
