"""Check a clean UC_AutoMaiL install with the Iris CLI and Python API."""

import argparse
from importlib.metadata import distribution
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import diy_automl
from diy_automl.excel_report_validation import validate_excel_report
from diy_automl.validation import validate


def check_extraction_and_training(cli, csv, output, features, storage):
    columns = [*features[:2], "target"]
    recipe = output / "columns.json"
    recipe.write_text(json.dumps({"columns": columns, "format": "csv"}), encoding="utf-8")
    subprocess.run(
        [*cli, "extract-data", "--source", str(csv), "--recipe", recipe.name, "--output", "extracted"],
        cwd=output,
        check=True,
    )
    extracted = output / "extracted"
    record = validate(extracted, schema=17)
    if record["settings"]["columns"] != columns or set(record["artifacts"]) != {"data.csv"}:
        raise RuntimeError("CSV recipe columns or format differ")

    training = diy_automl.select_csv(extracted / "data.csv", "target", output / "configured",
                                     tree_max_depths={"Decision Tree": 3}, cpu_threads=1, forest_jobs=2,
                                     bootstrap_repeats=20)
    settings = validate(training, schema=3)["settings"]
    expected = {"tree_max_depths": {"Decision Tree": 3}, "cpu_threads": 1, "forest_jobs": 2}
    if {name: settings[name] for name in expected} != expected:
        raise RuntimeError("Tree depth or CPU settings differ")

    if storage:
        for form in ("parquet", "duckdb"):
            result = diy_automl.extract_data(csv, output / form, columns, output_format=form)
            saved = validate(result, schema=17)
            if saved["rows"] != record["rows"] or set(saved["artifacts"]) != {f"data.{form}"}:
                raise RuntimeError(f"Unexpected {form} extraction")
    print(f"Validated the JSON recipe and tree-depth/CPU options{' and storage outputs' if storage else ''}")


def check_kalman(installed, version):
    import uc_automail as uc_al

    public_file = Path(uc_al.__file__).resolve()
    if public_file != installed.locate_file("uc_automail/__init__.py").resolve() or uc_al.__version__ != version:
        raise RuntimeError(f"Expected {version} from the installed public API, got {uc_al.__version__} at {public_file}")
    if uc_al.ExtendedKalmanFilter is not diy_automl.ExtendedKalmanFilter:
        raise RuntimeError("The public and compatibility imports expose different EKF classes")

    identity = np.eye(2)
    truth = np.arange(1, 7)[:, None] * np.array([1.0, 0.5])
    observations = truth + np.array([[0.1, -0.1], [-0.1, 0.1], [np.nan, 0.0], [0.1, -0.1], [0.0, 0.1], [-0.1, 0]])
    controls = np.broadcast_to([1.0, 0.5], observations.shape)
    settings = dict(
        initial_mean=[0, 0], initial_covariance=identity,
        transition=lambda x, k, u: x + u, transition_jacobian=lambda x, k, u: identity,
        measurement=lambda x, k: x, measurement_jacobian=lambda x, k: identity,
        process_covariance=0.01 * identity, observation_covariance=0.1 * identity,
    )
    batch = uc_al.extended_kalman_filter(observations, controls=controls, **settings)
    tracker = uc_al.ExtendedKalmanFilter(**settings)
    series = [tracker.step(row, control=u) for row, u in zip(observations, controls, strict=True)]
    for name, expected in batch.items():
        np.testing.assert_allclose(np.stack([step[name] for step in series]), expected, atol=1e-13, rtol=1e-13)

    diagnostics = uc_al.filter_diagnostics(batch, observations, max_lag=2)
    consistency = uc_al.evaluate_filter_consistency(truth, batch)
    if diagnostics["ljung_box_status"] != ("missing_observations", "ok"):
        raise RuntimeError("Ljung–Box status did not distinguish incomplete and complete channels")
    np.testing.assert_array_equal(diagnostics["autocorrelation_pairs"][0], [3, 5])
    np.testing.assert_allclose(diagnostics["autocorrelation_reference_limit"], [np.nan, 1.95996398454 / np.sqrt(6)])
    if not np.isfinite(diagnostics["nis"]).all() or not np.isfinite(consistency["nees"]).all():
        raise RuntimeError("The installed diagnostic did not produce finite NIS and NEES")
    print("Validated the public import, controlled EKF steps, missing-data diagnostics, NIS and NEES")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Expected installed version")
    parser.add_argument("--csv", type=Path, required=True, help="Path to the repository's Iris CSV")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the audit and evaluation")
    parser.add_argument("--storage", action="store_true", help="Check Parquet and DuckDB with the storage extra installed")
    parser.add_argument("--kalman", action="store_true", help="Check the streaming EKF and expanded diagnostics")
    args = parser.parse_args()

    installed = distribution("uc-automail")
    package_file = Path(diy_automl.__file__).resolve()
    if package_file != installed.locate_file("diy_automl/__init__.py").resolve():
        raise RuntimeError(f"Expected an installed distribution, got {package_file}")
    if installed.version != args.version or diy_automl.__version__ != args.version:
        raise RuntimeError(f"Expected {args.version}, got metadata {installed.version} and code {diy_automl.__version__}")
    if args.kalman:
        check_kalman(installed, args.version)

    csv = args.csv.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True)
    cli = [sys.executable, "-I", "-X", "utf8", "-m", "uc_automail" if args.kalman else "diy_automl"]
    subprocess.run([*cli, "--version"], cwd=output, check=True)

    for command, name, schema, options in (
        ("audit-data", "audit", 15, []),
        ("select", "selection", 3, ["--progress", "both"]),
    ):
        subprocess.run(
            [*cli, command, "--csv", str(csv), "--target", "target", *options, "--output", name],
            cwd=output,
            check=True,
        )
        result = validate(output / name, schema=schema)
        if result["version"] != args.version:
            raise RuntimeError(f"Unexpected version in the {command} manifest: {result['version']}")

    with np.load(output / "selection" / "arrays.npz", allow_pickle=False) as arrays:
        correct = np.count_nonzero(arrays["predictions"] == arrays["truth"])
        print(f"Iris holdout: {result['selected_model']}, {correct}/{len(arrays['truth'])} correct")
    if result["settings"].get("progress") != "both":
        raise RuntimeError("The installed selection did not enable progress recording")
    subprocess.run(
        [*cli, "report", "--run", "selection", "--charts", "detail", "--output", "detailed"],
        cwd=output,
        check=True,
    )
    detailed = validate_excel_report(output / "detailed")
    figure_ids = {figure["id"] for figure in detailed["figures"]}
    if detailed.get("charts") != "detail" or not {"sl_fold_scores", "sl_cv_progress"} <= figure_ids:
        raise RuntimeError("The installed report did not produce the detailed CV figures")
    print(f"Validated recorded progress and the illustrated report with {detailed['figure_count']} figures")

    check_extraction_and_training(cli, csv, output, result["features"], args.storage)
    print(f"Validated the audit and evaluation in {output}")


if __name__ == "__main__":
    main()
