"""Validate saved artifacts without importing TensorFlow or retraining."""
from datetime import date
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from .calendar_comparison import COMPARISON_SHEETS

SHEETS={3:{"SL_CV_Folds","SL_CV_Mean","SL_CV_SD","SL_Test","SL_Bootstrap","SL_Test_Intervals","Summary"},
        4:{"UL_By_Seed","UL_Seed_Summary","UL_Seed_Pairs","UL_Pair_Summary","UL_Reference","Summary"}}


def validate_chronological_selection(manifest, arrays, sheets):
    from .chronological_selection import chronological_partitions, chronology

    if "observation_days" not in arrays or manifest.get("split", {}).get("mode") != "chronological":
        raise ValueError("Missing chronological selection evidence")
    days = arrays["observation_days"]
    clock = manifest["split"].get("clock")
    if (days.ndim != 1 or days.dtype.kind not in "iu" or (days < 1).any()
            or (days > date.max.toordinal()).any() or clock not in ("utc", "local_unspecified")):
        raise ValueError("Invalid observation dates")
    if manifest["settings"]["time_column"] in manifest["features"]:
        raise ValueError("Timestamp column was used as a feature")
    if not sheets["SL_Bootstrap"].empty or not sheets["SL_Test_Intervals"].empty:
        raise ValueError("Unexpected intervals for chronological selection")
    train, test, folds = chronological_partitions(days)
    if not (np.array_equal(arrays["train"], train) and np.array_equal(arrays["test"], test)):
        raise ValueError("Invalid chronological holdout indices")
    for index, (a, b) in enumerate(folds):
        if not (np.array_equal(arrays[f"fold_{index}_train"], train[a])
                and np.array_equal(arrays[f"fold_{index}_validation"], train[b])):
            raise ValueError("Invalid chronological CV indices")
    if manifest["split"] != chronology(days, clock, train, test, folds):
        raise ValueError("Chronological period summaries differ")
    if "calendar_rule" in manifest:
        validate_calendar_comparison(manifest, arrays, sheets, days, train, test)


def validate_calendar_comparison(manifest, arrays, sheets, days, train, test):
    from .calendar_comparison import calendar_comparison
    from .ul_subsample_validation import same_table

    names = {"observation_hours", "train_truth", "truth", "predictions", "calendar_predictions", "majority_predictions"}
    if not names.issubset(arrays.files):
        raise ValueError("Missing calendar comparison evidence")
    hours = arrays["observation_hours"]
    if hours.shape != days.shape or hours.dtype.kind != "i" or (hours < -1).any() or (hours > 23).any():
        raise ValueError("Invalid recorded observation hours")
    class_count = len(manifest["classes"])
    for name in ("train_truth", "truth", "predictions"):
        labels = arrays[name]
        size = len(train) if name == "train_truth" else len(test)
        if labels.shape != (size,) or labels.dtype.kind not in "iu" or ((labels < 0) | (labels >= class_count)).any():
            raise ValueError(f"Invalid encoded labels in {name}")
    expected, baseline_predictions, rule = calendar_comparison(arrays, class_count)
    if manifest["calendar_rule"] != rule:
        raise ValueError("Calendar rule summary differs")
    for name, predictions in baseline_predictions.items():
        if not np.array_equal(arrays[name], predictions):
            raise ValueError(f"Calendar comparison predictions differ: {name}")
    for name, table in expected.items():
        same_table(sheets[name], table, f"Calendar comparison table differs: {name}")


def validate(directory,schema=None,allow_failed=False):
    root=Path(directory)
    m=json.loads((root/"manifest.json").read_text())
    version=m["schema_version"]
    if schema is not None and schema!=version: raise ValueError("Unexpected schema")
    if version==17:
        from .data_extract_validation import validate_data_extract
        return validate_data_extract(root)
    if version in (15,16):
        from .data_audit_validation import validate_data_audit
        return validate_data_audit(root)
    if version==14:
        from .annealing_validation import validate_annealing_check
        return validate_annealing_check(root)
    if version==11:
        from .sl_uncertainty_audit_validation import validate_sl_uncertainty_audit
        return validate_sl_uncertainty_audit(root)
    if version in (10,12,13):
        from .sl_ranking_validation import validate_sl_ranking
        return validate_sl_ranking(root)
    if version==9:
        from .uncertainty_validation import validate_uncertainty
        return validate_uncertainty(root)
    if version==7:
        from .tuning_validation import validate_tuning
        return validate_tuning(root,allow_failed=allow_failed)
    if version==6:
        from .ul_subsample_validation import validate_ul_subsample
        return validate_ul_subsample(root,allow_failed=allow_failed)
    if version==5:
        from .dl_repeat_validation import validate_dl_repeat
        return validate_dl_repeat(root,allow_failed=allow_failed)
    if m["status"]!="completed": raise ValueError("Run did not complete")
    for name,digest in m["artifacts"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    sheets=pd.read_excel(root/"report.xlsx",sheet_name=None)
    if {k:len(v) for k,v in sheets.items()}!=m["sheets"]: raise ValueError("Sheet row counts differ")
    if version in SHEETS:
        expected = SHEETS[version]
        if version == 3 and m["settings"].get("time_column") is not None and "calendar_rule" in m:
            expected = expected | COMPARISON_SHEETS
        if set(sheets) != expected:
            raise ValueError("Sheet names differ")
    with np.load(root/"arrays.npz",allow_pickle=False) as arrays:
        if version==2:
            if "UL_Result" in sheets:
                u=sheets["UL_Result"]
                if len(u)!=54 or u.duplicated(["Model","Cluster"]).any(): raise ValueError("Invalid UL candidates")
            for prefix in ("sl","dl"):
                if f"{prefix}_train" in arrays and set(arrays[f"{prefix}_train"])&set(arrays[f"{prefix}_test"]):
                    raise ValueError("Train/test overlap")
            if "dl_validation" in arrays:
                if set(arrays['dl_validation'])&(set(arrays['dl_train'])|set(arrays['dl_test'])):
                    raise ValueError("Validation overlap")
        elif version==3:
            if len(sheets['SL_Test'])!=1 or len(sheets['SL_CV_Folds'])!=25: raise ValueError("Invalid selection dimensions")
            settings = m["settings"]
            if settings.get("time_column") is not None:
                validate_chronological_selection(m, arrays, sheets)
                return m
            if len(sheets['SL_Bootstrap'])!=settings['bootstrap_repeats']: raise ValueError("Bootstrap count")
            train,test=set(arrays['train']),set(arrays['test'])
            if train&test: raise ValueError("Test overlap")
            group_mode = ("group_column" if settings.get("group_column") is not None else
                          "identical_inputs" if settings.get("group_identical") else None)
            groups = None
            if group_mode:
                if "group_ids" not in arrays or m.get("split", {}).get("mode") != group_mode:
                    raise ValueError("Missing grouped selection evidence")
                groups = arrays["group_ids"]
                if (groups.ndim != 1 or groups.dtype.kind not in "iu" or (groups < 0).any()
                        or sorted(train | test) != list(range(len(groups)))
                        or len(train) != len(arrays["train"]) or len(test) != len(arrays["test"])):
                    raise ValueError("Invalid grouped selection row partition")
                train_groups, test_groups = set(groups[arrays["train"]]), set(groups[arrays["test"]])
                if train_groups & test_groups:
                    raise ValueError("Train/test group overlap")
                if (len(train_groups) != m["split"]["train_groups"] or
                        len(test_groups) != m["split"]["test_groups"]):
                    raise ValueError("Grouped selection counts differ")
                if group_mode == "group_column" and settings["group_column"] in m["features"]:
                    raise ValueError("Group column was used as a feature")
            validation=[]
            for i in range(5):
                a,b=set(arrays[f'fold_{i}_train']),set(arrays[f'fold_{i}_validation'])
                if a&b or (a|b)!=train: raise ValueError("Invalid CV indices")
                if groups is not None and set(groups[list(a)]) & set(groups[list(b)]):
                    raise ValueError("CV group overlap")
                validation.extend(b)
            if sorted(validation)!=sorted(train): raise ValueError("CV validation coverage")
        elif version==4:
            n=len(m['settings']['seeds'])
            expected={'UL_By_Seed':18*n,'UL_Seed_Summary':18,'UL_Seed_Pairs':18*n*(n-1)//2,
                      'UL_Pair_Summary':18,'UL_Reference':36,'Summary':9}
            if m['sheets']!=expected: raise ValueError("Invalid repeat dimensions")
            if not (sheets['UL_By_Seed'].Status=='completed').all(): raise ValueError("Incomplete seeds")
            if sheets['UL_By_Seed'].duplicated(['Seed','Model','Cluster']).any(): raise ValueError("Duplicate seeds")
        else: raise ValueError("Unknown schema")
    return m
