"""Full-data reference plus per-subset preprocessing and clustering, schema 6."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .clustering import fit_one, models
from .clustering_subsample import (MAX_FEATURES, MAX_FITS, MAX_ROWS, compare_assignments,
                                  configuration, overlap_tables, sample_plan, sample_summary)
from .data import load_csv
from .reporting import fail, prepare, save, summary

FIT_COLUMNS = ['Status','Observed Clusters','Silhouette','Silhouette_SD','Silhouette_SE',
               'Davies-Bouldin','Calinski-Harabasz','Target ARI','Target AMI','Target NMI',
               'Train Time (s)','Converged','Iterations','Warnings','Parameters','Error']
REFERENCE_COLUMNS = ['Reference Status','Reference ARI','Reference Shared Rows',
                     'Reference Clusters','Sample Clusters']


def fit_fields(row):
    row = dict(row)
    # Only GMM exposes converged_; avoid turning an absent flag into a success claim.
    if row['Model'] != 'GMM':
        row['Converged'] = np.nan
    for name in ['ARI','AMI','NMI']:
        row['Target '+name] = row.pop(name)
    return row


def subsample_clustering_csv(csv, target, output, sample_seeds=range(42,52), fraction=.8,
                             model_seed=42, clusters=range(2,11), min_overlap=10):
    seeds,fraction,model_seed,ks,min_overlap = configuration(
        sample_seeds,fraction,model_seed,clusters,min_overlap)
    settings = dict(csv=str(csv),target=target,sample_seeds=seeds,fraction=fraction,
                    model_seed=model_seed,clusters=ks,min_overlap=min_overlap,
                    limits=dict(rows=MAX_ROWS,features=MAX_FEATURES,candidate_fits=MAX_FITS),
                    sampling='uniform without replacement; sorted source indices',
                    scaling='StandardScaler independently fitted within each subset',methods=['ul'])
    out,manifest = prepare(output,6,settings)
    try:
        data = load_csv(csv,target)
        plan = sample_plan(data.X,seeds,fraction,ks,min_overlap)
        candidates = list(models(model_seed,ks))
        configurations = pd.DataFrame([
            {'Candidate':cid,'Model':name,'Cluster':k,
             'Model Seed':model_seed if name in ['KMeans','GMM'] else np.nan,
             'Parameters':repr(model.get_params())}
            for cid,(name,k,model) in enumerate(candidates)])
        reference = []
        samples = []
        for _,candidate in configurations.iterrows():
            row = {**dict.fromkeys(FIT_COLUMNS,np.nan),**candidate.to_dict(),
                   'Status':'not_run','Warnings':'','Error':''}
            reference.append(row)
        for seed in seeds:
            for row in reference:
                samples.append({**row,'Sample Seed':seed,**dict.fromkeys(REFERENCE_COLUMNS,np.nan)})
        sample_rows = [{'Sample Seed':seed,'Rows':len(plan[seed]),'Effective Fraction':len(plan[seed])/len(data.y),
                        'Status':'not_run','Error':''} for seed in seeds]
        indices = pd.DataFrame([{'Sample Seed':seed,'Position':pos,'Source Index':int(index)}
                                for seed in seeds for pos,index in enumerate(plan[seed])])
        arrays = {'features':data.X,'truth':data.y,
                  **{f'sample_{seed}_indices':index for seed,index in plan.items()}}
        manifest.update(classes=data.classes,features=data.feature_names,rows=len(data.y),
                        subset_rows=len(plan[seeds[0]]),candidate_count=len(candidates),
                        protocol='Subset-specific preprocessing; no model/seed selection; descriptive stability only')

        def checkpoint():
            result = pd.DataFrame(samples)
            pairs,pair_summary = overlap_tables(result,arrays,seeds,min_overlap,configurations)
            manifest['reference_states'] = [r['Status'] for r in reference]
            manifest['sample_states'] = [{'sample_seed':r['Sample Seed'],'status':r['Status']} for r in sample_rows]
            sheets = {'UL_Reference':pd.DataFrame(reference),'UL_By_Subsample':result,
                      'UL_Sample_Summary':sample_summary(result),'UL_Overlap_Pairs':pairs,
                      'UL_Overlap_Summary':pair_summary,'UL_Samples':pd.DataFrame(sample_rows),
                      'UL_Sample_Indices':indices,'UL_Settings':configurations,
                      'Summary':summary([
                          ('Purpose','Data omission stability of preprocessing and clustering'),
                          ('Sampling','Uniform without replacement; not bootstrap; no target stratification'),
                          ('Fraction',fraction),('Sample seeds',str(seeds)),('Model seed',model_seed),
                          ('Scaling','Independent fit within each subset; reference scaler uses all rows'),
                          ('Reference ARI','Compare reference and subset assignments on the same source rows'),
                          ('Pair ARI',f'Only shared source rows; minimum overlap={min_overlap}; single-cluster comparisons unscored'),
                          ('Uncertainty','Dependent pairs; descriptive SD only; no SEM/CI; no automatic selection'),
                          ('Scope','Fixed observed data; omission sensitivity is not external validation'),
                      ])}
            save(out,manifest,sheets,arrays,finalize=False)

        checkpoint()
        manifest['stage'] = 'reference_scaling'
        scaler = StandardScaler().fit(data.X)
        Z = scaler.transform(data.X)
        arrays.update(reference_scaler_mean=scaler.mean_,reference_scaler_scale=scaler.scale_)
        manifest['stage'] = 'reference_fitting'
        reference[0]['Status'] = 'running'
        checkpoint()
        for cid,(name,k,model) in enumerate(candidates):
            try:
                row,labels = fit_one(Z,data.y,name,k,model)
                reference[cid].update(fit_fields(row),Status='completed')
                arrays[f'reference_labels_{cid}'] = labels
            except Exception as error:
                reference[cid].update(Status='failed',Error=f'{type(error).__name__}: {error}')
                manifest.update(status='failed',error=reference[cid]['Error'])
                checkpoint()
                raise
        checkpoint()
        for pos,seed in enumerate(seeds):
            sample_rows[pos]['Status'] = 'running'
            manifest.update(stage='sample_scaling',active_sample_seed=seed)
            checkpoint()
            try:
                subset = plan[seed]
                scaler = StandardScaler().fit(data.X[subset])
                transformed = scaler.transform(data.X[subset])
                arrays[f'sample_{seed}_scaler_mean'] = scaler.mean_
                arrays[f'sample_{seed}_scaler_scale'] = scaler.scale_
            except Exception as error:
                sample_rows[pos].update(Status='failed',Error=f'{type(error).__name__}: {error}')
                manifest.update(status='failed',error=sample_rows[pos]['Error'])
                checkpoint()
                raise
            manifest['stage'] = 'sample_fitting'
            samples[pos*len(candidates)]['Status'] = 'running'
            checkpoint()
            for cid,(name,k,model) in enumerate(models(model_seed,ks)):
                index = pos*len(candidates)+cid
                try:
                    row,labels = fit_one(transformed,data.y[subset],name,k,model)
                    comparison = compare_assignments(np.arange(len(data.y)),arrays[f'reference_labels_{cid}'],
                                                     subset,labels,min_overlap)
                    samples[index].update(fit_fields(row),Status='completed',
                                          **{'Reference Status':comparison['Comparison Status'],
                                             'Reference ARI':comparison['Assignment ARI'],
                                             'Reference Shared Rows':comparison['Shared Rows'],
                                             'Reference Clusters':comparison['Left Clusters'],
                                             'Sample Clusters':comparison['Right Clusters']})
                    arrays[f'sample_{seed}_labels_{cid}'] = labels
                except Exception as error:
                    samples[index].update(Status='failed',Error=f'{type(error).__name__}: {error}')
                    sample_rows[pos].update(Status='failed',Error=samples[index]['Error'])
                    manifest.update(status='failed',error=samples[index]['Error'])
                    checkpoint()
                    raise
            sample_rows[pos]['Status'] = 'completed'
            checkpoint()
        manifest.update(status='completed',stage='completed')
        manifest.pop('active_sample_seed',None)
        checkpoint()
        return out
    except Exception as error:
        fail(out,manifest,error)
        raise
