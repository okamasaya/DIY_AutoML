from pathlib import Path
import hashlib,json,platform,importlib.metadata
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from . import __version__

def json_default(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    raise TypeError(type(value).__name__)

def write_json(path,value):
    # Escaping Unicode keeps manifests readable across system encodings.
    Path(path).write_text(json.dumps(value,indent=2,default=json_default,allow_nan=False)+"\n",encoding="ascii")

def source_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}

def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def prepare(output,schema,settings):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    manifest={"schema_version":schema,"status":"running","implementation":"reimplemented; historical source unavailable",
              "version":__version__,"python":platform.python_version(),"settings":settings,"source_hashes":source_hashes(),
              "dependencies":{n:importlib.metadata.version(n) for n in ("numpy","pandas","scipy","scikit-learn","openpyxl")}}
    manifest["created_utc"]=datetime.now(timezone.utc).isoformat()
    manifest["platform"]=platform.platform()
    if "dl" in settings.get("methods",[]):
        from .dl_contract import RESULT_CONTRACT
        manifest["dl_backend"]="tensorflow"
        manifest["dl_result_contract"]=RESULT_CONTRACT
        for name in ("tensorflow-cpu","keras"):
            try: manifest["dependencies"][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: manifest["dependencies"][name]="not installed"
    csv=settings.get("csv")
    if csv: manifest["input_sha256"]=file_sha256(csv)
    write_json(output/"manifest.json",manifest)
    return output,manifest

def save(output,manifest,sheets,arrays,finalize=True,*,extra_artifacts=()):
    with pd.ExcelWriter(output/"report.xlsx",engine="openpyxl") as writer:
        for name,frame in sheets.items(): frame.to_excel(writer,sheet_name=name,index=False)
    np.savez_compressed(output/"arrays.npz",**arrays)
    manifest["sheets"]={name:len(frame) for name,frame in sheets.items()}
    manifest["artifacts"]={name:file_sha256(output/name) for name in ("report.xlsx", "arrays.npz", *extra_artifacts)}
    if finalize and manifest["status"]=="running": manifest["status"]="completed"
    write_json(output/"manifest.json",manifest)
    return output

def fail(output,manifest,error):
    manifest.update(status="failed",error=f"{type(error).__name__}: {error}")
    write_json(output/"manifest.json",manifest)

def summary(rows): return pd.DataFrame(rows,columns=["Item","Value"])
