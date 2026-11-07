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
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=json_default,allow_nan=False)+"\n")

def source_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}

def prepare(output,schema,settings):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    manifest={"schema_version":schema,"status":"running","implementation":"reimplemented; historical source unavailable",
              "version":__version__,"python":platform.python_version(),"settings":settings,"source_hashes":source_hashes(),
              "dependencies":{n:importlib.metadata.version(n) for n in ("numpy","pandas","scipy","scikit-learn","openpyxl")}}
    manifest["created_utc"]=datetime.now(timezone.utc).isoformat()
    manifest["platform"]=platform.platform()
    if "dl" in settings.get("methods",[]):
        for name in ("tensorflow-cpu","keras"):
            try: manifest["dependencies"][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: manifest["dependencies"][name]="not installed"
    csv=settings.get("csv")
    if csv: manifest["input_sha256"]=hashlib.sha256(Path(csv).read_bytes()).hexdigest()
    write_json(output/"manifest.json",manifest)
    return output,manifest

def save(output,manifest,sheets,arrays,finalize=True):
    with pd.ExcelWriter(output/"report.xlsx",engine="openpyxl") as writer:
        for name,frame in sheets.items(): frame.to_excel(writer,sheet_name=name,index=False)
    np.savez_compressed(output/"arrays.npz",**arrays)
    manifest["sheets"]={name:len(frame) for name,frame in sheets.items()}
    manifest["artifacts"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (output/"report.xlsx",output/"arrays.npz")}
    if finalize and manifest["status"]=="running": manifest["status"]="completed"
    write_json(output/"manifest.json",manifest)
    return output

def fail(output,manifest,error):
    manifest.update(status="failed",error=f"{type(error).__name__}: {error}")
    write_json(output/"manifest.json",manifest)

def summary(rows): return pd.DataFrame(rows,columns=["Item","Value"])
