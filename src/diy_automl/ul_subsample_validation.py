"""Saved-result validation for schema 6, without fitting a clustering model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from .clustering import models
from .clustering_subsample import (MAX_FEATURES, MAX_FITS, MAX_ROWS, compare_assignments, configuration, overlap_tables,
                                   sample_plan, sample_summary)

SHEETS = {'UL_Reference','UL_By_Subsample','UL_Sample_Summary','UL_Overlap_Pairs',
          'UL_Overlap_Summary','UL_Samples','UL_Sample_Indices','UL_Settings','Summary'}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def same_table(actual,expected,message):
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True),expected.reset_index(drop=True),
                                      check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    except AssertionError as error:
        raise ValueError(message) from error


def validate_ul_subsample(directory,allow_failed=False):
    root = Path(directory)
    m = json.loads((root/'manifest.json').read_text())
    require(m['schema_version']==6,'Unexpected schema')
    require(m['status']=='completed' or (allow_failed and m['status']=='failed'),'Run did not complete')
    for name in ['report.xlsx','arrays.npz']:
        require(hashlib.sha256((root/name).read_bytes()).hexdigest()==m['artifacts'][name],f'Artifact checksum mismatch: {name}')
    sheets = pd.read_excel(root/'report.xlsx',sheet_name=None)
    require(set(sheets)==SHEETS,'Sheet names differ')
    require({k:len(v) for k,v in sheets.items()}==m['sheets'],'Sheet row counts differ')
    s=m['settings']
    seeds,fraction,model_seed,ks,min_overlap=configuration(s['sample_seeds'],s['fraction'],s['model_seed'],s['clusters'],s['min_overlap'])
    require(s['limits']==dict(rows=MAX_ROWS,features=MAX_FEATURES,candidate_fits=MAX_FITS),'Computation limits differ')
    candidates=list(models(model_seed,ks))
    count=len(candidates)
    require(m['candidate_count']==count,'Candidate count differs')
    configs=sheets['UL_Settings']
    expected_configs=pd.DataFrame([{'Candidate':i,'Model':name,'Cluster':k,
                                   'Model Seed':model_seed if name in ['KMeans','GMM'] else np.nan,
                                   'Parameters':repr(model.get_params())}
                                  for i,(name,k,model) in enumerate(candidates)])
    same_table(configs,expected_configs,'Model configurations differ')
    ref=sheets['UL_Reference'];results=sheets['UL_By_Subsample'];samples=sheets['UL_Samples']
    require(ref.Candidate.tolist()==list(range(count)),'Reference order differs')
    require(results.Candidate.tolist()==list(range(count))*len(seeds),'Subsample candidate order differs')
    require(results['Sample Seed'].tolist()==[seed for seed in seeds for _ in candidates],'Subsample seed order differs')
    require(samples['Sample Seed'].tolist()==seeds,'Sample metadata order differs')
    for frame in [ref,results]:
        require(set(frame.Status)<={'completed','failed','not_run'},'Unknown or running candidate state')
        for _,row in frame.iterrows():
            expected=configs.iloc[int(row.Candidate)]
            require(row.Model==expected.Model and row.Cluster==expected.Cluster,'Candidate identity differs')
            require(row.Parameters==expected.Parameters,'Fit parameters differ')
            if row.Status!='completed':
                require(row[['Target ARI','Target AMI','Target NMI','Silhouette']].isna().all(),'Incomplete candidate has metrics')
    states=ref.Status.tolist()+results.Status.tolist()
    require(m['reference_states']==ref.Status.tolist(),'Reference states differ')
    require(m['sample_states']==[{'sample_seed':seed,'status':state} for seed,state in zip(seeds,samples.Status)],'Sample states differ')
    require(set(samples.Status)<={'completed','failed','not_run'},'Unknown or running sample state')
    if m['status']=='completed':
        require(set(states)=={'completed'} and samples.Status.eq('completed').all(),'Incomplete completed run')
    else:
        require('error' in m,'Missing failure reason')
        if 'failed' in states:
            where=states.index('failed')
            require(states==['completed']*where+['failed']+['not_run']*(len(states)-where-1),'Invalid candidate failure order')
        else:
            require(m['stage'] in ['reference_scaling','sample_scaling'],'Unexplained failure state')
            if m['stage']=='reference_scaling':
                require(set(states)=={'not_run'} and samples.Status.eq('not_run').all(),'Fits attempted after reference scaling failure')
            else:
                sample_states=samples.Status.tolist()
                require(sample_states.count('failed')==1,'Missing failed scaling sample')
                pos=sample_states.index('failed')
                require(sample_states==['completed']*pos+['failed']+['not_run']*(len(seeds)-pos-1),'Invalid sample failure order')
                require(states==['completed']*(count*(pos+1))+['not_run']*(count*(len(seeds)-pos)),'Fits attempted after sample scaling failure')
        for frame in [ref,results,samples]:
            require(frame.loc[frame.Status=='failed','Error'].notna().all(),'Missing failure message')
    with np.load(root/'arrays.npz',allow_pickle=False) as stored:
        arrays={key:stored[key] for key in stored.files}
    X,y=arrays['features'],arrays['truth']
    require(X.shape==(m['rows'],len(m['features'])) and np.isfinite(X).all(),'Invalid feature data')
    require(y.shape==(m['rows'],) and np.array_equal(np.unique(y),np.arange(len(m['classes']))),'Invalid target')
    plan=sample_plan(X,seeds,fraction,ks,min_overlap)
    size=len(plan[seeds[0]])
    require(size==m['subset_rows'],'Subsample size differs')
    require(samples.Rows.eq(size).all() and np.allclose(samples['Effective Fraction'],size/len(y),rtol=0,atol=1e-12),'Subsample metadata differs')
    expected_indices=pd.DataFrame([{'Sample Seed':seed,'Position':pos,'Source Index':int(index)}
                                   for seed in seeds for pos,index in enumerate(plan[seed])])
    same_table(sheets['UL_Sample_Indices'],expected_indices,'Source index table differs')
    expected_keys={'features','truth',*[f'sample_{seed}_indices' for seed in seeds]}

    def scaler_valid(prefix,raw,required):
        keys=[prefix+'_scaler_mean',prefix+'_scaler_scale']
        present=[key in arrays for key in keys]
        require(present[0]==present[1] and (not required or present[0]),'Missing scaler')
        if present[0]:
            fitted=StandardScaler().fit(raw)
            require(arrays[keys[0]].shape==fitted.mean_.shape and arrays[keys[1]].shape==fitted.scale_.shape,'Scaler shape differs')
            require(np.allclose(arrays[keys[0]],fitted.mean_,rtol=1e-12,atol=1e-12) and
                    np.allclose(arrays[keys[1]],fitted.scale_,rtol=1e-12,atol=1e-12),'Scaler not fitted on declared rows')
            expected_keys.update(keys)

    def labels_valid(row,key,truth):
        if row.Status!='completed':
            require(key not in arrays,'Incomplete fit has labels')
            return
        expected_keys.add(key)
        require(key in arrays,'Completed fit missing labels')
        if row.Model=='GMM':
            require(pd.notna(row.Converged) and row.Converged in [True,False],'Missing GMM convergence flag')
        else:
            require(pd.isna(row.Converged),'Unsupported convergence flag')
        labels=arrays[key]
        require(labels.shape==truth.shape and np.issubdtype(labels.dtype,np.integer),'Invalid label shape or type')
        require(np.isin(labels,np.arange(int(row.Cluster))).all(),'Invalid cluster label')
        require(row['Observed Clusters']==len(np.unique(labels)),'Observed cluster count differs')
        for metric,value in [('Target ARI',adjusted_rand_score(truth,labels)),
                             ('Target AMI',adjusted_mutual_info_score(truth,labels)),
                             ('Target NMI',normalized_mutual_info_score(truth,labels))]:
            require(np.isclose(row[metric],value,rtol=0,atol=1e-12),'Target metric differs from labels')

    scaler_valid('reference',X,ref.Status.eq('completed').any())
    for _,row in ref.iterrows():
        labels_valid(row,f'reference_labels_{int(row.Candidate)}',y)
    for seed in seeds:
        subset=plan[seed]
        require(np.array_equal(arrays[f'sample_{seed}_indices'],subset),'Sample indices do not reproduce')
        group=results[results['Sample Seed']==seed]
        sample=samples.loc[samples['Sample Seed']==seed].iloc[0]
        if sample.Status=='completed':
            require(group.Status.eq('completed').all(),'Sample completed before its candidates')
        elif sample.Status=='not_run':
            require(group.Status.eq('not_run').all(),'Unrun sample has attempted candidates')
            require(f'sample_{seed}_scaler_mean' not in arrays,'Unrun sample has a scaler')
        else:
            require(group.Status.eq('failed').sum()==1 or m['stage']=='sample_scaling','Failed sample has no failed stage')
        scaler_valid(f'sample_{seed}',X[subset],group.Status.eq('completed').any() or sample.Status=='completed')
        for _,row in group.iterrows():
            cid=int(row.Candidate)
            labels_valid(row,f'sample_{seed}_labels_{cid}',y[subset])
            if row.Status=='completed':
                require(ref.iloc[cid].Status=='completed','Missing reference for a completed subsample')
                comparison=compare_assignments(np.arange(len(y)),arrays[f'reference_labels_{cid}'],
                                               subset,arrays[f'sample_{seed}_labels_{cid}'],min_overlap)
                require(row['Reference Status']==comparison['Comparison Status'],'Reference comparison state differs')
                for actual,key in [('Reference Shared Rows','Shared Rows'),('Reference Clusters','Left Clusters'),
                                   ('Sample Clusters','Right Clusters'),('Reference ARI','Assignment ARI')]:
                    require(np.isclose(row[actual],comparison[key],rtol=0,atol=1e-12,equal_nan=True),'Reference comparison differs')
            else:
                require(pd.isna(row['Reference ARI']),'Incomplete fit has reference ARI')
    require(set(arrays)==expected_keys,'Unexpected arrays or labels')
    pairs,pair_summary=overlap_tables(results,arrays,seeds,min_overlap,configs)
    same_table(sheets['UL_Overlap_Pairs'],pairs,'Overlap pair comparisons differ')
    same_table(sheets['UL_Overlap_Summary'],pair_summary,'Overlap pair summary differs')
    same_table(sheets['UL_Sample_Summary'],sample_summary(results),'Subsample summary differs')
    return m
