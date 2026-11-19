import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from .supervised import candidates, pipeline, metrics, METRICS

def choose(means, metric="F1"):
    values = means[metric].to_numpy()
    ties = np.flatnonzero(np.isclose(values, values.max(), atol=1e-12, rtol=0))
    return int(ties[0]), ties.tolist()

def select(X,y,seed=42,metric="F1",*,folds=None):
    if metric not in METRICS: raise ValueError("Unknown selection metric")
    if np.bincount(y).min() < 5: raise ValueError("Five samples per training class required for 5-fold CV")
    if folds is None:
        folds = list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y))
    rows=[]
    for name, model in candidates(seed):
        for fold,(train,validation) in enumerate(folds):
            estimator=pipeline(clone(model)).fit(X[train],y[train])
            rows.append({"Model":name,"Fold":fold,**metrics(y[validation],estimator.predict(X[validation]),np.unique(y))})
    scores=pd.DataFrame(rows)
    means=scores.groupby("Model",sort=False)[list(METRICS)].mean().reset_index()
    sd=scores.groupby("Model",sort=False)[list(METRICS)].std().reset_index()
    index,ties=choose(means,metric)
    name,model=candidates(seed)[index]
    fitted=pipeline(model).fit(X,y)
    return fitted,name,scores,means,sd,ties,folds

def bootstrap(y,pred,repeats=1000,confidence=.95,seed=43,classes=None,*,group_ids=None):
    if repeats<1 or not 0<confidence<1: raise ValueError("Invalid bootstrap settings")
    rng=np.random.default_rng(seed)
    y=np.asarray(y);pred=np.asarray(pred)
    if y.ndim!=1 or pred.shape!=y.shape or len(y)==0: raise ValueError("Invalid bootstrap observations")
    classes=np.unique(y) if classes is None else np.asarray(classes)
    if classes.ndim!=1 or len(classes)!=len(np.unique(classes)) or not np.isin(y,classes).all():
        raise ValueError("Invalid bootstrap classes")
    samples=[]
    if group_ids is None:
        groups=[np.flatnonzero(y==c) for c in classes if np.any(y==c)]
        for _ in range(repeats):
            idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
            samples.append(metrics(y[idx],pred[idx],classes))
    else:
        group_ids = np.asarray(group_ids)
        if group_ids.shape != y.shape:
            raise ValueError("Group IDs must match the held-out rows")
        groups, codes = np.unique(group_ids, return_inverse=True)
        if len(groups) < 2:
            raise ValueError("Group bootstrap requires at least two held-out groups")
        for _ in range(repeats):
            draw = rng.integers(len(groups), size=len(groups))
            counts = np.bincount(draw, minlength=len(groups))
            samples.append(metrics(y, pred, classes, sample_weight=counts[codes]))
    frame=pd.DataFrame(samples)
    alpha=(1-confidence)/2
    intervals=pd.DataFrame([{"Metric":m,"Estimate":metrics(y,pred,classes)[m],
                             "Lower":frame[m].quantile(alpha),"Upper":frame[m].quantile(1-alpha)} for m in METRICS])
    return frame,intervals
