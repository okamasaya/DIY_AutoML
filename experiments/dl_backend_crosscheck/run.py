"""Compare the reference-size MLP on TensorFlow and PyTorch CPU."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

from diy_automl.data import load_csv
from diy_automl.deep_learning import prepare_data
from diy_automl.dl_contract import HIDDEN_LAYERS, integer
from diy_automl.metric_uncertainty import interval_rows, resample_indices, score_samples
from diy_automl.reporting import fail, save, summary, write_json
from diy_automl.sl_uncertainty_audit import parse_offsets
from diy_automl.supervised import METRICS

from worker import BETAS, EPSILON, LEARNING_RATE


BACKENDS = ("tensorflow", "pytorch")
EPOCH_COLUMNS = ("Training Loss", "Training Accuracy", "Validation Loss", "Validation Accuracy")
WORKER = Path(__file__).with_name("worker.py")
CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "TF_ENABLE_ONEDNN_OPTS": "0",
    "TF_CPP_MIN_LOG_LEVEL": "2", "KERAS_BACKEND": "tensorflow",
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def make_plan(data, scenarios, split_seed, training_seeds, epochs, batch_size):
    prepared = prepare_data(data.X, data.y, split_seed)
    dimensions = (data.X.shape[1], *HIDDEN_LAYERS, len(data.classes))
    initial, orders = [], []
    for seed in training_seeds:
        rng = np.random.default_rng(seed)
        weights = []
        for inputs, outputs in zip(dimensions, dimensions[1:]):
            bound = np.sqrt(6 / (inputs + outputs))
            weights.extend([rng.uniform(-bound, bound, (inputs, outputs)).astype("float32"),
                            np.zeros(outputs, dtype="float32")])
        initial.append(weights)
        orders.append(np.stack([rng.permutation(prepared.train) for _ in range(epochs)]))
    shifted = np.empty((len(scenarios), len(prepared.test), data.X.shape[1]), dtype="float32")
    for index, scenario in enumerate(scenarios):
        features = data.X[prepared.test].copy()
        features[:, data.feature_names.index(scenario["feature"])] += scenario["offset"]
        shifted[index] = (features - prepared.mean) / prepared.scale
    scaled = prepared.scaled.astype("float32")
    if not np.isfinite(shifted).all() or not np.isfinite(scaled).all():
        raise ValueError("Non-finite Float32 inputs to the DL comparison")
    return {
        **prepared.arrays(), "target": data.y, "scaled": scaled, "shifted_test": shifted,
        "batch_orders": np.stack(orders), "batch_size": np.asarray(batch_size),
        "training_seeds": np.asarray(training_seeds),
        **{f"initial_weight_{k}": np.stack([case[k] for case in initial]) for k in range(8)},
    }


def identify(table, backend, seed):
    table.insert(0, "Training Seed", seed)
    table.insert(0, "Backend", backend)
    return table


def forward_comparison(seed, condition, tf_logits, torch_logits, tf_probabilities, torch_probabilities):
    return {
        "Training Seed": seed, "Shared Weights": condition,
        "Max Absolute Logit Difference": np.max(np.abs(tf_logits - torch_logits)),
        "Max Absolute Probability Difference": np.max(np.abs(tf_probabilities - torch_probabilities)),
        "Different Labels": np.count_nonzero(tf_logits.argmax(axis=1) != torch_logits.argmax(axis=1)),
    }


def report_tables(plan, result, manifest):
    truth = plan["target"][plan["test"]]
    classes = np.arange(len(manifest["classes"]))
    draws = resample_indices(truth, manifest["bootstrap_repeats"], manifest["bootstrap_seed"])
    confidence = manifest["confidence"]
    tables = {
        "Summary": summary([
            ("Test Rows", len(truth)),
            ("Training", "Shared initialization and batch order, native Adam implementations"),
            ("Test Intervals", "Fixed fitted model, fixed observed class counts, shared rows"),
            ("Backend Differences", "PyTorch minus TensorFlow, paired on final test rows"),
            ("Actual Systematic Bias", "Unmeasured. Offset scenarios show sensitivity to an assumed shift."),
        ]),
    }
    evidence = {"test_bootstrap_indices": draws, "truth": truth}
    score_rows, intervals, differences, history, numeric, offsets = [], [], [], [], [], []
    seed_differences, seed_delta_samples = [], []
    tf_result, torch_result = (result[backend] for backend in BACKENDS)
    for index, seed in enumerate(plan["training_seeds"].tolist()):
        scores, samples = [], []
        for backend, current in result.items():
            predictions = current["test_logits"][index].argmax(axis=1)
            point, _, bootstrap, _, _ = score_samples(truth, predictions, classes, draws)
            scores.append(point)
            samples.append(bootstrap)
            evidence[f"{backend}_{seed}_predictions"] = predictions
            evidence[f"{backend}_{seed}_bootstrap_scores"] = bootstrap
            score_rows.append({"Backend": backend, "Training Seed": seed, **dict(zip(METRICS, point))})
            intervals.append(identify(interval_rows(point, bootstrap, METRICS, confidence), backend, seed))
            frame = pd.DataFrame(current["history"][index], columns=EPOCH_COLUMNS)
            frame.insert(0, "Epoch", np.arange(1, len(frame) + 1))
            history.append(identify(frame, backend, seed))
            for offset_index, scenario in enumerate(manifest["shifts"]):
                changed = current["shift_predictions"][index, offset_index]
                shifted, _, shifted_draws, _, _ = score_samples(truth, changed, classes, draws)
                effect = interval_rows(shifted - point, shifted_draws - bootstrap, METRICS, confidence)
                effect = effect.rename(columns={"Estimate": "Delta"})
                effect["Feature"] = scenario["feature"]
                effect["Assumed Offset"] = scenario["offset"]
                effect["Changed Predictions"] = np.count_nonzero(changed != predictions)
                offsets.append(identify(effect, backend, seed))
        delta, delta_samples = scores[1] - scores[0], samples[1] - samples[0]
        seed_differences.append(delta)
        seed_delta_samples.append(delta_samples)
        table = interval_rows(delta, delta_samples, METRICS, confidence).rename(columns={"Estimate": "PyTorch - TensorFlow"})
        table.insert(0, "Training Seed", seed)
        table["Different Labels"] = np.count_nonzero(
            evidence[f"tensorflow_{seed}_predictions"] != evidence[f"pytorch_{seed}_predictions"],
        )
        differences.append(table)
        numeric.extend([
            forward_comparison(
                seed, "Initial", tf_result["initial_logits"][index], torch_result["initial_logits"][index],
                tf_result["initial_probabilities"][index], torch_result["initial_probabilities"][index],
            ),
            forward_comparison(
                seed, "Trained TensorFlow", tf_result["test_logits"][index], torch_result["transferred_logits"][index],
                tf_result["test_probabilities"][index], torch_result["transferred_probabilities"][index],
            ),
        ])
    tables.update({
        "Test_Scores": pd.DataFrame(score_rows), "Test_Intervals": pd.concat(intervals, ignore_index=True),
        "Backend_Differences": pd.concat(differences, ignore_index=True),
        "Mean_Seed_Difference": interval_rows(
            np.mean(seed_differences, axis=0), np.mean(seed_delta_samples, axis=0), METRICS, confidence,
        ).rename(columns={"Estimate": "PyTorch - TensorFlow"}),
        "Forward_Parity": pd.DataFrame(numeric), "Epoch_History": pd.concat(history, ignore_index=True),
    })
    if offsets:
        tables["Offset_Scenarios"] = pd.concat(offsets, ignore_index=True)
    return tables, evidence


def crosscheck(csv, target, output, split_seed, training_seeds, epochs, batch_size, shift):
    epochs = integer(epochs, "epochs", minimum=1)
    batch_size = integer(batch_size, "batch_size", minimum=1)
    seeds = [integer(seed, "training_seed", maximum=2**32 - 1) for seed in training_seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Training seeds must be distinct")
    input_hash = digest(csv)
    data = load_csv(csv, target)
    if digest(csv) != input_hash:
        raise ValueError("Input CSV changed while reading")
    scenarios = parse_offsets(shift, data.feature_names)
    plan = make_plan(data, scenarios, split_seed, seeds, epochs, batch_size)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "stage": "preparation", "experiment": "dl_backend_crosscheck",
        "csv": str(csv.resolve()), "input_sha256": input_hash, "target": target, "classes": data.classes,
        "features": data.feature_names, "hidden_layers": HIDDEN_LAYERS, "dtype": "float32",
        "split_seed": split_seed, "training_seeds": seeds, "epochs": epochs, "batch_size": batch_size,
        "optimizer": {"name": "native Adam", "learning_rate": LEARNING_RATE, "betas": BETAS, "epsilon": EPSILON},
        "shifts": scenarios, "systematic_bias_status": "unmeasured", "cpu_environment": CPU_ENV,
        "bootstrap_repeats": 1000, "bootstrap_seed": 43, "confidence": .95,
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {name: version(name) for name in ("tensorflow-cpu", "keras", "torch", "numpy", "scikit-learn")},
        "code_sha256": {path.name: digest(path) for path in (Path(__file__), WORKER)},
    }
    write_json(output / "manifest.json", manifest)
    try:
        plan_path = output / "plan.npz"
        np.savez_compressed(plan_path, **plan)
        result = {}
        for backend in BACKENDS:
            manifest["stage"] = backend
            write_json(output / "manifest.json", manifest)
            worker_output = output / (backend + ".npz")
            command = [sys.executable, "-B", str(WORKER), backend, str(plan_path), str(worker_output)]
            if backend == "pytorch":
                command.extend(["--transferred", str(output / "tensorflow.npz")])
            subprocess.run(command, check=True, env={**os.environ, **CPU_ENV})
            with np.load(worker_output, allow_pickle=False) as saved:
                result[backend] = {name: saved[name] for name in saved.files}
        manifest["stage"] = "evaluation"
        tables, arrays = report_tables(plan, result, manifest)
        save(output, manifest, tables, arrays, finalize=False)
        for name in ("plan.npz", "tensorflow.npz", "pytorch.npz"):
            manifest["artifacts"][name] = digest(output / name)
        manifest.update(status="completed", stage="completed")
        write_json(output / "manifest.json", manifest)
        print(output)
        print(tables["Test_Scores"][["Backend", "Training Seed", "F1"]].to_string(index=False))
        print(tables["Forward_Parity"].to_string(index=False))
    except Exception as error:
        fail(output, manifest, error)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--target", default="target")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shift", action="append", default=[], metavar="FEATURE=OFFSET")
    crosscheck(**vars(parser.parse_args()))
