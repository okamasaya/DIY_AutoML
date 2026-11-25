"""Check a clean DIY AutoML install with the Iris CLI workflow."""

import argparse
from importlib.metadata import distribution
from pathlib import Path
import subprocess
import sys

import numpy as np

import diy_automl
from diy_automl.validation import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Expected installed version")
    parser.add_argument("--csv", type=Path, required=True, help="Path to the repository's Iris CSV")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the audit and evaluation")
    args = parser.parse_args()

    installed = distribution("diy-automl")
    package_file = Path(diy_automl.__file__).resolve()
    if package_file != installed.locate_file("diy_automl/__init__.py").resolve():
        raise RuntimeError(f"Expected an installed distribution, got {package_file}")
    if installed.version != args.version or diy_automl.__version__ != args.version:
        raise RuntimeError(f"Expected {args.version}, got metadata {installed.version} and code {diy_automl.__version__}")

    csv = args.csv.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True)
    cli = [sys.executable, "-I", "-X", "utf8", "-m", "diy_automl"]
    subprocess.run([*cli, "--version"], cwd=output, check=True)

    for command, name, schema in (("audit-data", "audit", 15), ("select", "selection", 3)):
        subprocess.run(
            [*cli, command, "--csv", str(csv), "--target", "target", "--output", name],
            cwd=output,
            check=True,
        )
        result = validate(output / name, schema=schema)
        if result["version"] != args.version:
            raise RuntimeError(f"Unexpected version in the {command} manifest: {result['version']}")

    with np.load(output / "selection" / "arrays.npz", allow_pickle=False) as arrays:
        correct = np.count_nonzero(arrays["predictions"] == arrays["truth"])
        print(f"Iris holdout: {result['selected_model']}, {correct}/{len(arrays['truth'])} correct")
    print(f"Validated the audit and evaluation in {output}")


if __name__ == "__main__":
    main()
