"""Training-only SL search followed by one refit, one holdout test and conditional CI."""
from numbers import Real

import numpy as np
import pandas as pd

from .clustering_subsample import integer
from .data import load_csv,split_indices
from .reporting import fail,prepare,save,summary
from .run_progress import RunProgress, progress_mode
from .selection import bootstrap
from .sl_compute import limit_sl_threads,sl_compute_settings,sl_compute_summary
from .supervised import METRICS,metrics,pipeline,sl_model_options,tree_depth_settings,tree_depth_summary
from .tuning import MAX_FEATURES,MAX_ROWS,aggregate,configurations,fit_checked,search,search_spaces


def tune_csv(csv,target,output,seed=42,metric='F1',bootstrap_repeats=1000,bootstrap_seed=43,confidence=.95,
             *,tree_max_depth=None,tree_max_depths=None,cpu_threads=None,forest_jobs=None,progress="off"):
    seed=integer(seed,'seed')
    bootstrap_seed=integer(bootstrap_seed,'bootstrap_seed')
    bootstrap_repeats=integer(bootstrap_repeats,'bootstrap_repeats',1,10000)
    if metric not in METRICS or isinstance(confidence,bool) or not isinstance(confidence,Real) or not 0<confidence<1:
        raise ValueError('Invalid metric or confidence')
    depth_options=tree_depth_settings(tree_max_depth, tree_max_depths)
    compute_options=sl_compute_settings(cpu_threads, forest_jobs)
    settings=dict(csv=str(csv),target=target,seed=seed,metric=metric,bootstrap_repeats=bootstrap_repeats,
                  bootstrap_seed=bootstrap_seed,confidence=float(confidence),search_spaces=search_spaces(**depth_options),
                  folds=5,tie_atol=1e-12,tie_rtol=0,limits=dict(rows=MAX_ROWS,features=MAX_FEATURES),
                  **depth_options,**compute_options)
    if progress_mode(progress) != "off":
        settings["progress"] = progress
    model_options=sl_model_options(settings)
    out,m=prepare(output,7,settings)
    events = RunProgress(progress, out)
    try:
        events.record("load")
        data=load_csv(csv,target)
        if len(data.y)>MAX_ROWS or data.X.shape[1]>MAX_FEATURES:
            raise ValueError(f'Tuning supports at most {MAX_ROWS} rows and {MAX_FEATURES} features')
        events.record("split")
        train,test=split_indices(data.y,seed)
        config=configurations(seed, **model_options)
        m.update(classes=data.classes,features=data.feature_names,rows=len(data.y),
                 protocol=f'Declared {len(config)} configurations, shared training-only 5-fold CV, one selected holdout test',
                 interval_scope='Fixed fitted selected model and observed test class counts; excludes search and training variability')
        arrays={'features':data.X,'target':data.y,'train':train,'test':test,'truth':data.y[test]}
        sheets={'SL_Search_Settings':config,
                'SL_Search_Folds':pd.DataFrame(), 'SL_Search_Mean':pd.DataFrame(), 'SL_Search_SD':pd.DataFrame(),
                'SL_Test':pd.DataFrame(columns=['Candidate','Model',*METRICS]),
                'SL_Bootstrap':pd.DataFrame(columns=METRICS),
                'SL_Test_Intervals':pd.DataFrame(columns=['Metric','Estimate','Lower','Upper'])}
        def checkpoint(extra_artifacts=()):
            sheets['Summary']=summary([
                ('Purpose','Bounded hyperparameter comparison on training pool only'),
                ('Selection',f'Unweighted 5-fold mean {metric}; common folds and train-only scaling'),
                ('Ties','atol=1e-12, rtol=0; declared candidate order; no time/test tiebreak'),
                ('Configurations',len(config)), ('Test','One selected configuration only'),
                ('Search uncertainty','Selection CV is optimistic after search; not an unbiased performance estimate'),
                ('Interval',m['interval_scope']),
                ('Zero width','Does not imply zero unknown uncertainty'),
                ('Failure','Stop on first fold/refit failure or ConvergenceWarning; no candidate skipping'),
                ('Data','Bundled datasets are reused development data, not unseen external validation'),
                *tree_depth_summary(settings),
                *sl_compute_summary(settings),
            ])
            save(out,m,sheets,arrays,finalize=False,extra_artifacts=extra_artifacts)
        def save_search_progress(rows,local):
            m['stage']='cv'
            sheets['SL_Search_Folds']=pd.DataFrame(rows)
            sheets['SL_Search_Mean'],sheets['SL_Search_SD']=aggregate(rows)
            for key,value in local.items():
                if key.endswith('_train') or key.endswith('_validation'):
                    value=train[value]
                arrays[key]=value
            checkpoint()
        events.record("cv_start")
        with limit_sl_threads(settings):
            selected,cid,ties,folds,means,sd,local=search(
                data.X[train], data.y[train], seed, metric, save_search_progress,
                events=events if progress != "off" else None, **model_options,
            )
        name,model,_=selected
        events.record("refit")
        m.update(selected_candidate=cid,selected_model=name,tied_candidates=ties,stage='refit')
        checkpoint()
        fitted=pipeline(model)
        with limit_sl_threads(settings):
            m['refit_warnings']=fit_checked(fitted,data.X[train],data.y[train])
        arrays.update(selected_scaler_mean=fitted[0].mean_,selected_scaler_scale=fitted[0].scale_)
        events.record("holdout")
        m['stage']='test'
        checkpoint()
        with limit_sl_threads(settings):
            predictions=fitted.predict(data.X[test])
        arrays['predictions']=predictions
        sheets['SL_Test']=pd.DataFrame([{'Candidate':cid,'Model':name,**metrics(data.y[test],predictions,np.unique(data.y))}])
        events.record("bootstrap")
        m['stage']='bootstrap'
        checkpoint()
        samples,intervals=bootstrap(data.y[test],predictions,bootstrap_repeats,confidence,bootstrap_seed,classes=np.unique(data.y))
        sheets['SL_Bootstrap']=samples;sheets['SL_Test_Intervals']=intervals
        events.record("report")
        m.update(status='completed',stage='completed')
        checkpoint(events.attach(sheets))
        return out
    except Exception as error:
        events.failed()
        fail(out,m,error)
        raise
