from .data import load_csv
from .reporting import prepare,save,fail,summary
from .sl_compute import limit_sl_threads, sl_compute_settings, sl_compute_summary
from . import supervised,clustering

def run_csv(csv,target,output,methods=("sl","ul","dl"),seed=42,epochs=100,split_seed=None,training_seed=None,
            *, tree_max_depth=None, tree_max_depths=None, cpu_threads=None, forest_jobs=None):
    if isinstance(methods,str): methods=(methods,)
    if not methods or set(methods)-{"sl","ul","dl"}: raise ValueError("Unknown methods")
    settings = dict(csv=str(csv),target=target,methods=list(methods),seed=seed,epochs=epochs,
                    split_seed=split_seed,training_seed=training_seed)
    depth_options = supervised.tree_depth_settings(tree_max_depth, tree_max_depths)
    compute_options = sl_compute_settings(cpu_threads, forest_jobs)
    if depth_options and "sl" not in methods:
        raise ValueError("Tree depth settings require the sl method")
    if compute_options and "sl" not in methods:
        raise ValueError("CPU settings require the sl method")
    settings.update(depth_options)
    settings.update(compute_options)
    model_options = supervised.sl_model_options(settings)
    out,m=prepare(output,2,settings)
    try:
        d=load_csv(csv,target); sheets={}; arrays={"features":d.X,"target":d.y}
        m.update(classes=d.classes,features=d.feature_names,
            protocol={"sl":"Exploratory fixed holdout; scaler fitted only on training samples",
                      "ul":"Full-data structural analysis; 54 candidates, no automatic selection",
                      "dl":"MLP32-16-8 relu/softmax; Adam .001; sparse categorical crossentropy; batch16; deterministic CPU; training-seeded dataset shuffle"},
            sl_models={name:repr(model.get_params()) for name,model in
                       supervised.candidates(seed, **model_options)})
        for method in methods:
            if method=="dl":
                from .deep_learning import run
                result,history,a=run(d.X,d.y,seed,split_seed,training_seed,epochs)
                sheets["DL_Result"]=result; sheets["DL_History"]=history
            else:
                if method=="sl":
                    with limit_sl_threads(settings):
                        result,a=supervised.run(d.X,d.y,seed,**model_options)
                else:
                    result,a=clustering.run(d.X,d.y,seed)
                sheets[f"{method.upper()}_Result"]=result
            arrays.update({f"{method}_{k}":v for k,v in a.items()})
        sheets["Summary"]=summary([("Purpose","Exploratory comparison; no combined SL/UL/DL ranking"),
                    ("UL uncertainty","Pointwise Silhouette SD/SE are not seed variability or confidence intervals"),
                    ("Data","Bundled datasets are development data, not unseen external validation"),
                    *supervised.tree_depth_summary(settings), *sl_compute_summary(settings)])
        return save(out,m,sheets,arrays)
    except Exception as e: fail(out,m,e); raise
