import numpy as np
import pandas as pd
from .calendar_comparison import COMPARISON, calendar_comparison
from .chronological_selection import chronological_partitions, chronology, require_class_coverage
from .data import load_csv,split_indices
from .grouped_selection import grouped_partitions, identical_input_groups
from .selection import select,bootstrap
from .supervised import METRICS, metrics
from .reporting import prepare,save,fail,summary,file_sha256,write_json

def select_csv(csv,target,output,seed=42,metric="F1",bootstrap_repeats=1000,bootstrap_seed=43,confidence=.95,
               *, expected_input_sha256=None, bp_summary=False, group_column=None, group_identical=False,
               time_column=None):
    if group_column is not None and group_identical:
        raise ValueError("Choose a group column or group identical inputs")
    if time_column is not None and (group_column is not None or group_identical):
        raise ValueError("Choose chronological or grouped selection")
    settings = dict(csv=str(csv),target=target,seed=seed,metric=metric)
    if time_column is None:
        settings.update(bootstrap_repeats=bootstrap_repeats,bootstrap_seed=bootstrap_seed,confidence=confidence)
    else:
        settings["time_column"] = time_column
    if bp_summary:
        settings["bp_summary"] = True
    if group_column is not None:
        settings["group_column"] = group_column
    if group_identical:
        settings["group_identical"] = True
    out,m=prepare(output,3,settings)
    try:
        if expected_input_sha256 is not None and m["input_sha256"] != expected_input_sha256:
            raise ValueError("The training CSV no longer matches its audit")
        d=load_csv(csv,target,target_as_text=bp_summary,group_column=group_column,time_column=time_column)
        if expected_input_sha256 is not None and file_sha256(csv) != expected_input_sha256:
            raise ValueError("The training CSV changed while it was being read")
        groups = identical_input_groups(d.X) if group_identical else d.groups
        inner_folds, test_groups = None, None
        interval_scope = "Conditional on fixed fitted model and observed class counts; excludes training and selection variability"
        split_rows = []
        if d.observation_days is not None:
            days = d.observation_days
            train, test, inner_folds = chronological_partitions(days)
            require_class_coverage(d.y, train, test, inner_folds)
            m["split"] = chronology(days, d.clock, train, test, inner_folds)
            interval_scope = "Point estimates for the holdout period. Temporal uncertainty interval pending."
            split_rows = [("Split", "Most recent 20% of observed dates held out (rounded up). Five expanding windows."),
                          ("Timestamp column", time_column), ("Clock", d.clock)]
            for key, title in (("training", "Training period"), ("test", "Held-out period")):
                period = m["split"][key]
                split_rows.append((title, f"{period['start']} through {period['end']} "
                                          f"({period['days']} days, {period['rows']} rows)"))
            for index, fold in enumerate(m["split"]["folds"], start=1):
                fit, review = fold["training"], fold["validation"]
                split_rows.append((f"Window {index} (SL_CV_Folds {index - 1})",
                                   f"Train from {fit['start']} through {fit['end']}. "
                                   f"Validate from {review['start']} through {review['end']}."))
        elif groups is None:
            train,test=split_indices(d.y,seed)
        else:
            train, test, inner_folds, holdout = grouped_partitions(d.y, groups, seed)
            test_groups = groups[test]
            mode = "identical_inputs" if group_identical else "group_column"
            m["split"] = {
                "mode": mode, "outer_fold": holdout,
                "train_groups": len(np.unique(groups[train])), "test_groups": len(np.unique(test_groups)),
            }
            interval_scope = "Conditional on fixed fitted model and observed test groups; resamples whole groups and excludes training and selection variability"
            split_rows = [
                ("Split", f"Group-disjoint holdout and five-fold CV ({mode})"),
                ("Training groups", m["split"]["train_groups"]), ("Held-out groups", m["split"]["test_groups"]),
                ("Holdout rows", len(test)), ("Outer fold", holdout),
            ]
        model,name,folds,means,sd,ties,indices=select(d.X[train],d.y[train],seed,metric,folds=inner_folds)
        pred=model.predict(d.X[test])
        test_scores = metrics(d.y[test], pred, np.unique(d.y))
        if d.observation_days is None:
            samples,intervals=bootstrap(d.y[test],pred,bootstrap_repeats,confidence,bootstrap_seed,
                                        classes=np.unique(d.y),group_ids=test_groups)
        else:
            samples = pd.DataFrame(columns=METRICS)
            intervals = pd.DataFrame(columns=["Metric", "Estimate", "Lower", "Upper"])
        m["selected_parameters"]=repr(model.get_params())
        m.update(classes=d.classes,features=d.feature_names,selected_model=name,tied_candidates=means.iloc[ties].Model.tolist(),
                 interval_scope=interval_scope)
        arrays={"train":train,"test":test,"predictions":pred,"truth":d.y[test],
                "scaler_mean":model[0].mean_,"scaler_scale":model[0].scale_}
        if groups is not None:
            arrays["group_ids"] = groups
        if d.observation_days is not None:
            arrays["observation_days"] = d.observation_days
            arrays["observation_hours"] = d.observation_hours
            arrays["train_truth"] = d.y[train]
        for i,(a,b) in enumerate(indices): arrays[f"fold_{i}_train"]=train[a]; arrays[f"fold_{i}_validation"]=train[b]
        sheets={"SL_CV_Folds":folds,"SL_CV_Mean":means,"SL_CV_SD":sd,
                "SL_Test":pd.DataFrame([{"Model":name,**test_scores}]),
                "SL_Bootstrap":samples,"SL_Test_Intervals":intervals,
                "Summary":summary([("Selected",name),("Selection",f"5-fold mean {metric}; atol=1e-12, rtol=0; candidate order"),
                    ("Interval",m["interval_scope"]),("Zero width","Does not imply no unknown uncertainty"), *split_rows])}
        if d.observation_days is not None:
            calendar_sheets, baselines, rule = calendar_comparison(arrays, len(d.classes))
            sheets.update(calendar_sheets)
            arrays.update(baselines)
            m["calendar_rule"] = rule
            sources = rule["source_rows"]
            sheets["Summary"] = pd.concat([sheets["Summary"], summary([
                ("Calendar rule", "Training majority for the same weekday and hour. Fall back to weekday, then overall."),
                ("Calendar rule rows", f"Weekday and hour {sources['weekday_hour']}, weekday {sources['weekday']}, "
                                       f"overall {sources['overall']}"),
                ("Monthly macro F1", "Uses the full label set. An undefined class F1 contributes zero."),
            ])], ignore_index=True)
        if not bp_summary:
            return save(out,m,sheets,arrays)

        from .bp_evaluation import BP_REPORT, summarize_evaluation, write_bp_report

        result = summarize_evaluation(d, arrays, test_scores, intervals, m, sheets.get(COMPARISON))
        save(out,m,sheets,arrays,finalize=False)
        write_bp_report(out, result, sheets)
        m["bp_summary"] = result
        m["artifacts"][BP_REPORT] = file_sha256(out / BP_REPORT)
        m["status"] = "completed"
        write_json(out / "manifest.json", m)
        return out
    except Exception as e: fail(out,m,e); raise
