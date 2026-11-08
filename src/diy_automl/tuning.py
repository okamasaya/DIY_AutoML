"""A declared, bounded SL grid; all decisions use only the training pool."""
from itertools import product
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold

from .selection import choose
from .supervised import METRICS, candidates, metrics, pipeline

MAX_ROWS = 5000
MAX_FEATURES = 100
SEARCH_SPACES = [
    {'max_depth':[None,3,5], 'min_samples_leaf':[1,3]},
    {'max_depth':[None,5], 'min_samples_leaf':[1,3]},
    {'C':[1.,.1,10.]},
    {'kernel':['rbf','linear'], 'C':[1.,.1,10.]},
    {'n_neighbors':[5,3,7,9], 'weights':['uniform','distance']},
]
FOLD_COLUMNS = ['Candidate','Model','Fold','Status',*METRICS,'Fit Time (s)','Warnings','Error']


def search_candidates(seed=42):
    result=[]
    for (name,base),space in zip(candidates(seed),SEARCH_SPACES):
        for values in product(*space.values()):
            changed=dict(zip(space,values))
            estimator=clone(base).set_params(**changed)
            result.append((name,estimator,changed))
    return result


def configurations(seed=42):
    return pd.DataFrame([
        {'Candidate':i,'Model':name,'Search Parameters':json.dumps(changed,sort_keys=True),
         'Estimator Parameters':json.dumps(model.get_params(),sort_keys=True)}
        for i,(name,model,changed) in enumerate(search_candidates(seed))])


class ConvergenceFailure(RuntimeError):
    pass


def fit_checked(estimator,X,y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        estimator.fit(X,y)
    messages='; '.join(str(w.message) for w in caught)
    if any(issubclass(w.category,ConvergenceWarning) for w in caught):
        raise ConvergenceFailure('Tuning stopped on ConvergenceWarning: '+messages)
    return messages


def aggregate(rows):
    means=[];deviations=[]
    frame=pd.DataFrame(rows,columns=FOLD_COLUMNS)
    for candidate,group in frame.groupby('Candidate',sort=False):
        done=group[group.Status=='completed']
        state=('completed' if len(done)==5 else 'failed' if group.Status.eq('failed').any()
               else 'running' if group.Status.isin(['running','completed']).any() else 'not_run')
        identity={'Candidate':candidate,'Model':group.Model.iloc[0],'Status':state,'Completed Folds':len(done)}
        means.append({**identity,**{m:done[m].mean() if len(done)==5 else np.nan for m in METRICS}})
        deviations.append({**identity,**{m:done[m].std(ddof=1) if len(done)==5 else np.nan for m in METRICS}})
    return pd.DataFrame(means),pd.DataFrame(deviations)


def search(X,y,seed=42,metric='F1',progress=None):
    """The caller passes the training pool only. This function has no test input."""
    if metric not in METRICS:
        raise ValueError('Unknown selection metric')
    if np.bincount(y).min()<5:
        raise ValueError('Five samples per training class required for 5-fold CV')
    folds=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y))
    if min(len(train) for train,_ in folds)<9:
        raise ValueError('The declared k-NN grid requires at least 9 training samples per fold')
    grid=search_candidates(seed)
    rows=[{'Candidate':i,'Model':name,'Fold':fold,'Status':'not_run',
           **dict.fromkeys(METRICS,np.nan),'Fit Time (s)':np.nan,'Warnings':'','Error':''}
          for i,(name,_,_) in enumerate(grid) for fold in range(5)]
    arrays={}
    for fold,(train,validation) in enumerate(folds):
        arrays[f'fold_{fold}_train']=train
        arrays[f'fold_{fold}_validation']=validation
        arrays[f'fold_{fold}_truth']=y[validation]
    def checkpoint():
        if progress is not None: progress(rows,arrays)
    checkpoint()
    for cid,(name,base,_) in enumerate(grid):
        rows[cid*5]['Status']='running'
        checkpoint()
        for fold,(train,validation) in enumerate(folds):
            row=rows[cid*5+fold]
            started=time.perf_counter()
            try:
                fitted=pipeline(clone(base))
                row['Warnings']=fit_checked(fitted,X[train],y[train])
                elapsed=time.perf_counter()-started
                pred=fitted.predict(X[validation])
                if cid==0:
                    arrays[f'fold_{fold}_scaler_mean']=fitted[0].mean_
                    arrays[f'fold_{fold}_scaler_scale']=fitted[0].scale_
                else:
                    if not (np.array_equal(arrays[f'fold_{fold}_scaler_mean'],fitted[0].mean_)
                            and np.array_equal(arrays[f'fold_{fold}_scaler_scale'],fitted[0].scale_)):
                        raise RuntimeError('Shared fold preprocessing differs between candidates')
                arrays[f'candidate_{cid}_fold_{fold}_predictions']=pred
                row.update(metrics(y[validation],pred,np.unique(y)),Status='completed',**{'Fit Time (s)':elapsed})
            except Exception as error:
                row.update(Status='failed',Error=f'{type(error).__name__}: {error}',
                           **{'Fit Time (s)':time.perf_counter()-started})
                checkpoint()
                raise
        checkpoint()
    means,sd=aggregate(rows)
    selected,ties=choose(means,metric)
    return grid[selected],selected,ties,pd.DataFrame(rows),means,sd,arrays
