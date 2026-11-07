from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from .clustering import models,fit_one

class RepeatFailure(RuntimeError):
    def __init__(self,message,result): super().__init__(message); self.result=result

def run(X,y,seeds=range(42,52),ks=range(2,11)):
    seeds=list(seeds); ks=list(ks)
    if len(seeds)<2 or len(set(seeds))!=len(seeds): raise ValueError("At least two unique seeds required")
    if len(X)>5000: raise ValueError("UL supports at most 5000 rows")
    scaler=StandardScaler().fit(X); Z=scaler.transform(X)
    arrays={"scaler_mean":scaler.mean_,"scaler_scale":scaler.scale_,"truth":y}
    rows=[]; reference=[]; pairs=[]; labels={}; error=None
    for seed in seeds:
        for name,k,model in models(seed,ks,True):
            row={"Seed":seed,"Model":name,"Cluster":k,"Status":"not_run","Parameters":repr(model.get_params())}
            if error is None:
                try:
                    fitted,assignment=fit_one(Z,y,name,k,model)
                    row.update(fitted,Status="completed")
                    row["Target ARI"]=row.pop("ARI")
                    labels[seed,name,k]=assignment
                    arrays[f"labels_{seed}_{name}_{k}"]=assignment
                except Exception as e:
                    row.update(Status="failed",Error=f"{type(e).__name__}: {e}"); error=e
            rows.append(row)
    for name,k,model in models(seeds[0],ks):
        if not name.startswith("Hierarchical"): continue
        row={"Model":name,"Cluster":k,"Status":"not_run","Parameters":repr(model.get_params())}
        if error is None:
            try:
                fitted,assignment=fit_one(Z,y,name,k,model); row.update(fitted,Status="completed")
                arrays[f"reference_{name}_{k}"]=assignment
            except Exception as e:
                row.update(Status="failed",Error=f"{type(e).__name__}: {e}"); error=e
        reference.append(row)
    for a,b in combinations(seeds,2):
        for name in ("KMeans","GMM"):
            for k in ks:
                if (a,name,k) in labels and (b,name,k) in labels:
                    pairs.append({"Model":name,"Cluster":k,"Seed A":a,"Seed B":b,
                                  "Assignment ARI":adjusted_rand_score(labels[a,name,k],labels[b,name,k])})
    by_seed=pd.DataFrame(rows)
    pair_frame=pd.DataFrame(pairs,columns=["Model","Cluster","Seed A","Seed B","Assignment ARI"])
    completed=by_seed[by_seed.Status=="completed"]
    metrics=[c for c in ("Target ARI","Silhouette","Davies-Bouldin","Calinski-Harabasz") if c in completed]
    seed_summary=(completed.groupby(["Model","Cluster"])[metrics].agg(["count","mean","std","min","max"]).reset_index()
                  if metrics else pd.DataFrame(columns=["Model","Cluster"]))
    seed_summary.columns=["_".join(str(x) for x in c if x!="") if isinstance(c,tuple) else c for c in seed_summary.columns]
    pair_summary=pair_frame.groupby(["Model","Cluster"])["Assignment ARI"].agg(["count","mean","std","min","max"]).reset_index()
    result=({"UL_By_Seed":by_seed,"UL_Seed_Summary":seed_summary,"UL_Seed_Pairs":pair_frame,
             "UL_Pair_Summary":pair_summary,"UL_Reference":pd.DataFrame(reference)},arrays)
    if error is not None: raise RepeatFailure(str(error),result)
    return result
