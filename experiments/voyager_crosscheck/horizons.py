"""Compare forecasts using JPL Horizons trajectories as reference positions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from run import ENGINES, OBSERVER, SEED, SENSORS, prepare, report, run_engine, write_json


SOURCE_BODIES = {"voyager1": "-31", "earth": "399"}
FIRST_JD = 2444574.5
QUERY_OPTIONS = {
    "format": "json", "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'VECTORS'",
    "CENTER": "'500@10'", "START_TIME": "'1980-12-01'", "STOP_TIME": "'1982-11-29'", "STEP_SIZE": "'7 d'",
    "REF_PLANE": "'ECLIPTIC'", "REF_SYSTEM": "'ICRF'", "OUT_UNITS": "'AU-D'", "VEC_TABLE": "'2'",
    "CSV_FORMAT": "'YES'", "VEC_CORR": "'NONE'", "TIME_TYPE": "'TDB'",
}


def query_url(body):
    settings = {**QUERY_OPTIONS, "COMMAND": f"'{SOURCE_BODIES[body]}'"}
    return "https://ssd.jpl.nasa.gov/api/horizons.api?" + urlencode(settings)


def read_vectors(path, body, days):
    raw = path.read_bytes()
    payload = json.loads(raw)
    result = payload.get("result", "")
    header, start, vector_table = result.partition("$$SOE")
    table, end, _ = vector_table.partition("$$EOE")
    if payload.get("error") or not start or not end:
        raise ValueError(f"Horizons did not return a vector table for {body}")
    target = next((line for line in header.splitlines() if line.startswith("Target body name:")), "")
    required = ("Sun (10)", "AU-D", "GEOMETRIC cartesian states", "Ecliptic of J2000.0")
    if f"({SOURCE_BODIES[body]})" not in target or any(token not in header for token in required):
        raise ValueError(f"Unexpected body or vector frame for {body}")
    rows = list(csv.reader(table.strip().splitlines()))
    times = np.array([float(row[0]) for row in rows])
    vectors = np.array([[float(value) for value in row[2:8]] for row in rows])
    if not np.array_equal(times, FIRST_JD + days) or vectors.shape != (len(days), 6) or not np.isfinite(vectors).all():
        raise ValueError(f"Unexpected Horizons vector schedule for {body}")
    return vectors, {
        "query": query_url(body), "target": target, "signature": payload["signature"],
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def prepare_horizons(source, run):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from make_voyager_demo import (BIAS_SIGMA, BIAS_START_DAY, DAYS_PER_YEAR, OBSERVATION_STD,
                                   PRIOR_MEAN, STATE_COLUMNS, TIME_DAYS, TRUE_START, measurement)

    voyager, voyager_source = read_vectors(source / "voyager1.json", "voyager1", TIME_DAYS)
    earth, earth_source = read_vectors(source / "earth.json", "earth", TIME_DAYS)
    first_position, first_velocity = np.split(voyager[0], 2)
    along = first_position / np.linalg.norm(first_position)
    normal = np.cross(first_position, first_velocity)
    normal /= np.linalg.norm(normal)
    plane = np.column_stack((along, np.cross(normal, along)))
    states = np.column_stack((voyager[:, :3] @ plane, voyager[:, 3:] @ plane * DAYS_PER_YEAR))
    observer = np.column_stack((earth[1:, :3] @ plane, earth[1:, 3:] @ plane * DAYS_PER_YEAR))
    ideal = np.array([measurement(state, site) for state, site in zip(states[1:], observer, strict=True)])
    nominal = ideal + np.random.default_rng(SEED).normal(0, OBSERVATION_STD, ideal.shape)
    bias = nominal.copy()
    days = TIME_DAYS[1:]
    bias[days >= BIAS_START_DAY] += BIAS_SIGMA * OBSERVATION_STD

    samples = run.with_name(run.name + "-samples")
    samples.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(states, columns=STATE_COLUMNS).assign(time_day=TIME_DAYS).to_csv(samples / "truth.csv", index=False)
    observations = [pd.DataFrame(np.column_stack((observer, values)), columns=OBSERVER + SENSORS)
                    .assign(time_day=days, scenario=scenario)
                    for scenario, values in (("nominal", nominal), ("bias", bias))]
    pd.concat(observations, ignore_index=True).to_csv(samples / "observations.csv", index=False)
    prepare(samples, run)
    write_json(run / "horizons.json", {
        "sources": {"voyager1": voyager_source, "earth": earth_source},
        "first_date_tdb": "1980-12-01", "last_date_tdb": "1982-11-29", "rows": len(TIME_DAYS),
        "projection": "Plane defined by the first Voyager heliocentric position and velocity",
        "max_out_of_plane_distance_au": float(np.max(np.abs(voyager[:, :3] @ normal))),
        "observations": "Simulated range and range rate within the projected plane",
        "normal_noise_sd": dict(zip(SENSORS, OBSERVATION_STD.tolist(), strict=True)),
        "observation_seed": SEED,
        "prior_mean": (states[0] + PRIOR_MEAN - TRUE_START).tolist(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("urls", "prepare", *ENGINES, "report"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--source", type=Path, help="Folder containing voyager1.json and earth.json from the urls action")
    args = parser.parse_args()
    if args.action == "urls":
        for body in SOURCE_BODIES:
            print(query_url(body))
        return
    if args.run is None:
        parser.error("--run is required for this action")
    if args.action == "prepare":
        if args.source is None:
            parser.error("--source is required for prepare")
        prepare_horizons(args.source, args.run)
    else:
        data = pd.read_csv(args.run / "dataset.csv")
        if args.action == "report":
            report(args.run, data)
        else:
            context = json.loads((args.run / "horizons.json").read_text(encoding="utf-8"))
            run_engine(args.action, args.run, data, initial_mean=context["prior_mean"])
    print(args.run)


if __name__ == "__main__":
    main()
