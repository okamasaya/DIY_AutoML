"""Create a verifiable backup without Git metadata, credentials, or virtualenvs."""
import argparse
from pathlib import Path
import hashlib,json,zipfile

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
if a.output.exists(): raise FileExistsError(a.output)
a.output.parent.mkdir(parents=True,exist_ok=True)
files=[]
for name in ['.gitattributes','README.md','pyproject.toml','requirements-local.lock.txt','.gitignore','src','scripts','tests','docs','data','notebooks','originals','validation']:
    path=root/name
    files.extend([path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()])
files=[p for p in files if '__pycache__' not in p.parts and not any(part.endswith('.egg-info') for part in p.parts)]
files.extend((root/'dist').glob('*.whl'))
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(files): z.write(path,'DIY_AutoML/'+str(path.relative_to(root)))
    z.writestr('DIY_AutoML/BACKUP_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(a.output) as z:
    if z.testzip() is not None: raise ValueError('ZIP CRC failure')
    for name,digest in manifest.items():
        if hashlib.sha256(z.read('DIY_AutoML/'+name)).hexdigest()!=digest: raise ValueError(name)
digest=hashlib.sha256(a.output.read_bytes()).hexdigest()
a.output.with_suffix(a.output.suffix+'.sha256').write_text(digest+'  '+a.output.name+'\n')
print(f'{a.output}\nSHA256 {digest}\n{len(files)} files + backup manifest')
