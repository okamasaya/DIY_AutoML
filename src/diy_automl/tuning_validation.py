"""Validate schema 7 using saved predictions and indices, without classifier refitting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .data import split_indices
from .selection import bootstrap,choose
from .supervised import METRICS,metrics
from .tuning import MAX_FEATURES,MAX_ROWS,SEARCH_SPACES,aggregate,configurations
from .ul_subsample_validation import require,same_table

SHEETS={'SL_Search_Settings','SL_Search_Folds','SL_Search_Mean','SL_Search_SD',
        'SL_Test','SL_Bootstrap','SL_Test_Intervals','Summary'}


def validate_tuning(directory,allow_failed=False):
    root=Path(directory)
    m=json.loads((root/'manifest.json').read_text())
    require(m['schema_version']==7,'Unexpected schema')
    require(m['status']=='completed' or (allow_failed and m['status']=='failed'),'Run did not complete')
    for name in ['report.xlsx','arrays.npz']:
        require(hashlib.sha256((root/name).read_bytes()).hexdigest()==m['artifacts'][name],f'Artifact checksum mismatch: {name}')
    sheets=pd.read_excel(root/'report.xlsx',sheet_name=None)
    require(set(sheets)==SHEETS,'Sheet names differ')
    require({key:len(value) for key,value in sheets.items()}==m['sheets'],'Sheet row counts differ')
    s=m['settings'];seed=s['seed'];metric=s['metric']
    require(s['search_spaces']==SEARCH_SPACES and s['folds']==5,'Search space or folds differ')
    require(s['limits']==dict(rows=MAX_ROWS,features=MAX_FEATURES),'Limits differ')
    require(s['tie_atol']==1e-12 and s['tie_rtol']==0,'Tie rule differs')
    configs=configurations(seed)
    same_table(sheets['SL_Search_Settings'],configs,'Configurations differ')
    rows=sheets['SL_Search_Folds']
    require(rows.Candidate.tolist()==[i for i in range(27) for _ in range(5)],'Candidate fold order differs')
    require(rows.Fold.tolist()==list(range(5))*27,'Fold order differs')
    require(set(rows.Status)<={'completed','failed','not_run'},'Unknown or running fold status')
    for _,row in rows.iterrows():
        require(row.Model==configs.iloc[int(row.Candidate)].Model,'Model identity differs')
        if row.Status!='completed': require(row[list(METRICS)].isna().all(),'Incomplete fold has metrics')
    means,sd=aggregate(rows.to_dict('records'))
    same_table(sheets['SL_Search_Mean'],means,'Mean scores differ')
    same_table(sheets['SL_Search_SD'],sd,'Fold SD differs')
    states=rows.Status.tolist()
    if m['status']=='completed':
        require(set(states)=={'completed'},'Incomplete completed search')
    else:
        require('error' in m,'Missing failure reason')
        if 'failed' in states:
            i=states.index('failed')
            require(states==['completed']*i+['failed']+['not_run']*(135-i-1),'Invalid fold failure order')
            require(m['stage']=='cv' and pd.notna(rows.iloc[i].Error),'Missing CV failure details')
        else:
            require(set(states)=={'completed'} and m['stage'] in ['refit','test','bootstrap'],'Unexplained failure')
    with np.load(root/'arrays.npz',allow_pickle=False) as stored:
        arrays={key:stored[key] for key in stored.files}
    X,y=arrays['features'],arrays['target']
    require(X.shape==(m['rows'],len(m['features'])) and np.isfinite(X).all(),'Invalid feature data')
    require(len(y)==len(X) and np.array_equal(np.unique(y),np.arange(len(m['classes']))),'Invalid target')
    train,test=split_indices(y,seed)
    require(np.array_equal(arrays['train'],train) and np.array_equal(arrays['test'],test),'Holdout split differs')
    require(np.array_equal(arrays['truth'],y[test]),'Test truth differs')
    folds=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(X[train],y[train]))
    expected_keys={'features','target','train','test','truth'}
    for fold,(a,b) in enumerate(folds):
        for name,value in [('train',train[a]),('validation',train[b]),('truth',y[train[b]])]:
            key=f'fold_{fold}_{name}'
            require(np.array_equal(arrays[key],value),'Shared CV indices or truth differ')
            expected_keys.add(key)
        completed=rows[(rows.Fold==fold)&(rows.Status=='completed')]
        scaler_keys=[f'fold_{fold}_scaler_mean',f'fold_{fold}_scaler_scale']
        if len(completed):
            scaler=StandardScaler().fit(X[train[a]])
            for key,value in zip(scaler_keys,[scaler.mean_,scaler.scale_]):
                require(key in arrays and arrays[key].shape==value.shape and np.allclose(arrays[key],value,rtol=1e-12,atol=1e-12),'Fold scaler differs from training rows')
                expected_keys.add(key)
        for _,row in completed.iterrows():
            key=f'candidate_{int(row.Candidate)}_fold_{fold}_predictions'
            require(key in arrays,'Completed fold missing predictions')
            prediction=arrays[key]
            require(prediction.shape==y[train[b]].shape and np.isin(prediction,np.unique(y)).all(),'Invalid CV predictions')
            for name,estimate in metrics(y[train[b]],prediction,np.unique(y)).items():
                require(np.isclose(row[name],estimate,rtol=0,atol=1e-12),'CV metric differs from predictions')
            expected_keys.add(key)
    if 'selected_candidate' in m:
        require(set(states)=={'completed'},'Selection before all folds completed')
        cid,ties=choose(means,metric)
        require(cid==m['selected_candidate'] and ties==m['tied_candidates'],'Training-only selection differs')
        require(m['selected_model']==configs.iloc[cid].Model,'Selected model differs')
    else:
        require(m['status']=='failed' and m['stage']=='cv','Missing selection')
    selected_scaler_keys=['selected_scaler_mean','selected_scaler_scale']
    if any(key in arrays for key in selected_scaler_keys):
        scaler=StandardScaler().fit(X[train])
        for key,value in zip(selected_scaler_keys,[scaler.mean_,scaler.scale_]):
            require(key in arrays and arrays[key].shape==value.shape and np.allclose(arrays[key],value,rtol=1e-12,atol=1e-12),'Selected scaler differs from training pool')
            expected_keys.add(key)
    if 'predictions' in arrays:
        require(all(key in arrays for key in selected_scaler_keys),'Test without refit scaler')
        pred=arrays['predictions']
        require(pred.shape==y[test].shape and np.isin(pred,np.unique(y)).all(),'Invalid test predictions')
        expected=pd.DataFrame([{'Candidate':m['selected_candidate'],'Model':m['selected_model'],**metrics(y[test],pred,np.unique(y))}])
        same_table(sheets['SL_Test'],expected,'Test score differs from predictions')
        expected_keys.add('predictions')
    else:
        require(sheets['SL_Test'].empty,'Test score without predictions')
        require(m['status']=='failed','Missing test predictions')
    if m['status']=='completed':
        require(m['stage']=='completed','Completed status with unfinished stage')
        samples,intervals=bootstrap(y[test],arrays['predictions'],s['bootstrap_repeats'],s['confidence'],s['bootstrap_seed'],classes=np.unique(y))
        same_table(sheets['SL_Bootstrap'],samples,'Conditional bootstrap samples differ')
        same_table(sheets['SL_Test_Intervals'],intervals,'Conditional bootstrap intervals differ')
    else:
        require(sheets['SL_Bootstrap'].empty and sheets['SL_Test_Intervals'].empty,'Unexpected completed intervals in failed run')
    require(set(arrays)==expected_keys,'Unexpected arrays or predictions from uncompleted fits')
    return m
