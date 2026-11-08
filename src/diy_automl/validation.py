"""Validate saved artifacts without importing TensorFlow or retraining."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

SHEETS={3:{"SL_CV_Folds","SL_CV_Mean","SL_CV_SD","SL_Test","SL_Bootstrap","SL_Test_Intervals","Summary"},
        4:{"UL_By_Seed","UL_Seed_Summary","UL_Seed_Pairs","UL_Pair_Summary","UL_Reference","Summary"}}

def validate(directory,schema=None,allow_failed=False):
    root=Path(directory)
    m=json.loads((root/"manifest.json").read_text())
    version=m["schema_version"]
    if schema is not None and schema!=version: raise ValueError("Unexpected schema")
    if version==9:
        from .uncertainty_validation import validate_uncertainty
        return validate_uncertainty(root)
    if version==7:
        from .tuning_validation import validate_tuning
        return validate_tuning(root,allow_failed=allow_failed)
    if version==6:
        from .ul_subsample_validation import validate_ul_subsample
        return validate_ul_subsample(root,allow_failed=allow_failed)
    if version==5:
        from .dl_repeat_validation import validate_dl_repeat
        return validate_dl_repeat(root,allow_failed=allow_failed)
    if m["status"]!="completed": raise ValueError("Run did not complete")
    for name,digest in m["artifacts"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    sheets=pd.read_excel(root/"report.xlsx",sheet_name=None)
    if {k:len(v) for k,v in sheets.items()}!=m["sheets"]: raise ValueError("Sheet row counts differ")
    if version in SHEETS and set(sheets)!=SHEETS[version]: raise ValueError("Sheet names differ")
    with np.load(root/"arrays.npz",allow_pickle=False) as arrays:
        if version==2:
            if "UL_Result" in sheets:
                u=sheets["UL_Result"]
                if len(u)!=54 or u.duplicated(["Model","Cluster"]).any(): raise ValueError("Invalid UL candidates")
            for prefix in ("sl","dl"):
                if f"{prefix}_train" in arrays and set(arrays[f"{prefix}_train"])&set(arrays[f"{prefix}_test"]):
                    raise ValueError("Train/test overlap")
            if "dl_validation" in arrays:
                if set(arrays['dl_validation'])&(set(arrays['dl_train'])|set(arrays['dl_test'])):
                    raise ValueError("Validation overlap")
        elif version==3:
            if len(sheets['SL_Test'])!=1 or len(sheets['SL_CV_Folds'])!=25: raise ValueError("Invalid selection dimensions")
            if len(sheets['SL_Bootstrap'])!=m['settings']['bootstrap_repeats']: raise ValueError("Bootstrap count")
            train,test=set(arrays['train']),set(arrays['test'])
            if train&test: raise ValueError("Test overlap")
            validation=[]
            for i in range(5):
                a,b=set(arrays[f'fold_{i}_train']),set(arrays[f'fold_{i}_validation'])
                if a&b or (a|b)!=train: raise ValueError("Invalid CV indices")
                validation.extend(b)
            if sorted(validation)!=sorted(train): raise ValueError("CV validation coverage")
        elif version==4:
            n=len(m['settings']['seeds'])
            expected={'UL_By_Seed':18*n,'UL_Seed_Summary':18,'UL_Seed_Pairs':18*n*(n-1)//2,
                      'UL_Pair_Summary':18,'UL_Reference':36,'Summary':9}
            if m['sheets']!=expected: raise ValueError("Invalid repeat dimensions")
            if not (sheets['UL_By_Seed'].Status=='completed').all(): raise ValueError("Incomplete seeds")
            if sheets['UL_By_Seed'].duplicated(['Seed','Model','Cluster']).any(): raise ValueError("Duplicate seeds")
        else: raise ValueError("Unknown schema")
    return m
