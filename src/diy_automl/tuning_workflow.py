"""Training-only SL search followed by one refit, one holdout test and conditional CI."""
from numbers import Real

import numpy as np
import pandas as pd

from .clustering_subsample import integer
from .data import load_csv,split_indices
from .reporting import fail,prepare,save,summary
from .selection import bootstrap
from .supervised import METRICS,metrics,pipeline
from .tuning import MAX_FEATURES,MAX_ROWS,SEARCH_SPACES,aggregate,configurations,fit_checked,search


def tune_csv(csv,target,output,seed=42,metric='F1',bootstrap_repeats=1000,bootstrap_seed=43,confidence=.95):
    seed=integer(seed,'seed')
    bootstrap_seed=integer(bootstrap_seed,'bootstrap_seed')
    bootstrap_repeats=integer(bootstrap_repeats,'bootstrap_repeats',1,10000)
    if metric not in METRICS or isinstance(confidence,bool) or not isinstance(confidence,Real) or not 0<confidence<1:
        raise ValueError('Invalid metric or confidence')
    settings=dict(csv=str(csv),target=target,seed=seed,metric=metric,bootstrap_repeats=bootstrap_repeats,
                  bootstrap_seed=bootstrap_seed,confidence=float(confidence),search_spaces=SEARCH_SPACES,
                  folds=5,tie_atol=1e-12,tie_rtol=0,limits=dict(rows=MAX_ROWS,features=MAX_FEATURES))
    out,m=prepare(output,7,settings)
    try:
        data=load_csv(csv,target)
        if len(data.y)>MAX_ROWS or data.X.shape[1]>MAX_FEATURES:
            raise ValueError(f'Tuning supports at most {MAX_ROWS} rows and {MAX_FEATURES} features')
        train,test=split_indices(data.y,seed)
        m.update(classes=data.classes,features=data.feature_names,rows=len(data.y),
                 protocol='Declared 27 configurations, shared training-only 5-fold CV, one selected holdout test',
                 interval_scope='Fixed fitted selected model and observed test class counts; excludes search and training variability')
        arrays={'features':data.X,'target':data.y,'train':train,'test':test,'truth':data.y[test]}
        config=configurations(seed)
        sheets={'SL_Search_Settings':config,
                'SL_Search_Folds':pd.DataFrame(), 'SL_Search_Mean':pd.DataFrame(), 'SL_Search_SD':pd.DataFrame(),
                'SL_Test':pd.DataFrame(columns=['Candidate','Model',*METRICS]),
                'SL_Bootstrap':pd.DataFrame(columns=METRICS),
                'SL_Test_Intervals':pd.DataFrame(columns=['Metric','Estimate','Lower','Upper'])}
        def checkpoint():
            sheets['Summary']=summary([
                ('Purpose','Bounded hyperparameter comparison on training pool only'),
                ('Selection',f'Unweighted 5-fold mean {metric}; common folds and train-only scaling'),
                ('Ties','atol=1e-12, rtol=0; declared candidate order; no time/test tiebreak'),
                ('Configurations',27), ('Test','One selected configuration only'),
                ('Search uncertainty','Selection CV is optimistic after search; not an unbiased performance estimate'),
                ('Interval',m['interval_scope']),
                ('Zero width','Does not imply zero unknown uncertainty'),
                ('Failure','Stop on first fold/refit failure or ConvergenceWarning; no candidate skipping'),
                ('Data','Bundled datasets are reused development data, not unseen external validation'),
            ])
            save(out,m,sheets,arrays,finalize=False)
        def progress(rows,local):
            m['stage']='cv'
            sheets['SL_Search_Folds']=pd.DataFrame(rows)
            sheets['SL_Search_Mean'],sheets['SL_Search_SD']=aggregate(rows)
            for key,value in local.items():
                if key.endswith('_train') or key.endswith('_validation'):
                    value=train[value]
                arrays[key]=value
            checkpoint()
        selected,cid,ties,folds,means,sd,local=search(data.X[train],data.y[train],seed,metric,progress)
        name,model,_=selected
        m.update(selected_candidate=cid,selected_model=name,tied_candidates=ties,stage='refit')
        checkpoint()
        fitted=pipeline(model)
        m['refit_warnings']=fit_checked(fitted,data.X[train],data.y[train])
        arrays.update(selected_scaler_mean=fitted[0].mean_,selected_scaler_scale=fitted[0].scale_)
        m['stage']='test'
        checkpoint()
        predictions=fitted.predict(data.X[test])
        arrays['predictions']=predictions
        sheets['SL_Test']=pd.DataFrame([{'Candidate':cid,'Model':name,**metrics(data.y[test],predictions,np.unique(data.y))}])
        m['stage']='bootstrap'
        checkpoint()
        samples,intervals=bootstrap(data.y[test],predictions,bootstrap_repeats,confidence,bootstrap_seed,classes=np.unique(data.y))
        sheets['SL_Bootstrap']=samples;sheets['SL_Test_Intervals']=intervals
        m.update(status='completed',stage='completed')
        checkpoint()
        return out
    except Exception as error:
        fail(out,m,error)
        raise
