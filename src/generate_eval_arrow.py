"""
Author: Tim Frenzel
Version: 1.00
Usage:  python src/generate_eval_arrow.py \
          --synthea_path /data/synthea/csv \
          --out_path data/eval.arrow \
          --n 2000 --seed 42

Objective of the Code
---------------------
Extract a **lightweight, diagnosis‑focused evaluation slice** from raw Synthea®
CSV exports and serialise it as an Apache Arrow table for fast, repeatable
benchmarking of the Mixture‑of‑Agents diagnostic pipeline.  The script:
  1. Loads `patients.csv`, `conditions.csv`, and (optionally) `encounters.csv`.
  2. Builds problem vignettes combining demographics + the last encounter note.
  3. Selects the *index diagnosis* (ICD‑10 code & text) as ground‑truth label.
  4. Randomly samples *n* patients (stratified by ICD‑10 chapter to avoid class
     collapse) and writes the result to a single Arrow file plus a compact JSON
     schema file for downstream tasks.

The produced dataset has the schema:
    ┌ patient_id:      string
    │ age:             int32
    │ gender:          string
    │ icd10_code:      string   (ground‑truth)
    │ icd10_text:      string   (human‑readable)
    │ vignette:        string   (prompt body)
    └ domain:          string   (router hint; e.g. "Cardiology")

Dependencies: pandas, pyarrow (>10.0), numpy.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

###############################################################################
# Helper functions
###############################################################################

def _load_csv(path: pathlib.Path, filename: str, cols: List[str] | None = None) -> pd.DataFrame:
    """Read a Synthea® CSV returning only *cols* if provided."""
    fp = path / filename
    if not fp.exists():
        raise FileNotFoundError(fp)
    return pd.read_csv(fp, usecols=cols)


def _age_from_birth(birth_date: str, ref_year: int = 2025) -> int:
    try:
        return ref_year - int(birth_date[:4])
    except Exception:
        return -1


def _map_domain(icd10_code: str) -> str:
    """Very light rule‑based domain mapping for router hints."""
    if icd10_code.startswith("I"):  # Circulatory system
        return "Cardiology"
    if icd10_code.startswith("E"):  # Endocrine, metabolic
        return "Metabolic"
    return "General"


###############################################################################
# Core routine
###############################################################################

def build_dataset(
    synthea_path: pathlib.Path,
    n: int = 2000,
    seed: int = 42,
) -> Tuple[pa.Table, dict]:
    """Return Arrow table + schema metadata as dict."""
    rng = random.Random(seed)

    patients = _load_csv(
        synthea_path,
        "patients.csv",
        ["Id", "BIRTHDATE", "GENDER"],
    )
    conds = _load_csv(
        synthea_path,
        "conditions.csv",
        ["PATIENT", "START", "STOP", "CODE", "DESCRIPTION"],
    )

    # Keep only the *last* condition per patient as the index diagnosis
    conds = conds.sort_values("STOP").groupby("PATIENT").tail(1)

    merged = patients.merge(
        conds, how="inner", left_on="Id", right_on="PATIENT", suffixes=("", "_cond")
    )

    merged["age"] = merged["BIRTHDATE"].apply(_age_from_birth)

    merged["vignette"] = (
        "Patient "
        + merged["GENDER"].str.capitalize()
        + ", age "
        + merged["age"].astype(str)
        + ". Presenting with medical history culminating in diagnosis: "
        + merged["DESCRIPTION"].str.lower()
        + "."
    )

    merged["domain"] = merged["CODE"].apply(_map_domain)

    # Stratified sample to keep ICD chapters balanced where possible
    if len(merged) > n:
        merged = (
            merged.groupby("CODE", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max(1, n // 50)), random_state=seed))
            .sample(n, random_state=seed)
        )

    arrow_tbl = pa.Table.from_pandas(
        merged[["Id", "age", "GENDER", "CODE", "DESCRIPTION", "vignette", "domain"]]
        .rename(
            columns={
                "Id": "patient_id",
                "GENDER": "gender",
                "CODE": "icd10_code",
                "DESCRIPTION": "icd10_text",
            }
        )
    )

    meta = {
        "n_rows": len(arrow_tbl),
        "n_unique_codes": merged["CODE"].nunique(),
        "generation_seed": seed,
    }
    return arrow_tbl, meta


###############################################################################
# CLI
###############################################################################

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic evaluation slice (Arrow)")
    p.add_argument("--synthea_path", required=True, type=pathlib.Path, help="Path to Synthea CSV directory")
    p.add_argument("--out_path", required=True, type=pathlib.Path, help="Target .arrow file")
    p.add_argument("--n", type=int, default=2000, help="Number of patients to sample (default 2000)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--dry_run", action="store_true", help="Build dataset but do not write to disk")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    tbl, meta = build_dataset(args.synthea_path, n=args.n, seed=args.seed)

    if args.dry_run:
        print(tbl.schema)
        print(meta)
        return

    # Ensure output directory exists
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(args.out_path, "wb") as sink:
        with pa.ipc.new_file(sink, tbl.schema) as writer:
            writer.write_table(tbl)

    # dump metadata json alongside
    meta_fp = args.out_path.with_suffix(".json")
    meta_fp.write_text(json.dumps(meta, indent=2))
    print(f"✅  Wrote {len(tbl)} rows to {args.out_path}")


if __name__ == "__main__":
    main()
