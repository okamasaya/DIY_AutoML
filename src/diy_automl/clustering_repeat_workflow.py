from .data import load_csv
from .clustering_repeated import run,RepeatFailure
from .reporting import prepare,save,fail,summary

def repeat_clustering_csv(csv,target,output,seeds=range(42,52)):
    seeds=list(seeds)
    out,m=prepare(output,4,dict(csv=str(csv),target=target,seeds=seeds))
    try:
        d=load_csv(csv,target)
        try: sheets,arrays=run(d.X,d.y,seeds)
        except RepeatFailure as e:
            sheets,arrays=e.result; m.update(status="failed",error=str(e))
        m.update(classes=d.classes,features=d.feature_names)
        sheets["Summary"]=summary([("Purpose","Initialization sensitivity on fixed data"),("Seeds",str(seeds)),
            ("Scaling","Shared StandardScaler fitted on full dataset"),("KMeans","n_init=auto"),
            ("GMM","n_init=1; covariance_type=full"),("Hierarchy","One reference fit per candidate"),
            ("Target ARI","Agreement with target labels"),("Assignment ARI","Pairwise agreement between seeds"),
            ("Uncertainty","Seed pairs are dependent; no SEM/CI; no automatic selection")])
        save(out,m,sheets,arrays)
        if m["status"]=="failed": raise RuntimeError(m["error"])
        return out
    except Exception as e: fail(out,m,e); raise
