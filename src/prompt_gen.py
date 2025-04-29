#!/usr/bin/env python
"""
Author: Tim Frenzel
Version: 1.6
Usage:  Imported by other scripts (e.g., build_corpus.py, generate_eval_arrow.py).
        Can be run directly for preview: `python src/prompt_gen.py --preview 5`

Objective of the Code:
------------
Contains the core logic for generating patient diagnostic prompts. It queries 
a DuckDB database (containing structured Synthea data), extracts patient demographics 
and historical diagnoses up to a cutoff date, and formats this information into a 
natural language prompt. It also identifies the first diagnosis after the cutoff 
as the prediction target (label).

Changes v1.5 → v1.6
-------------------
✓ Casts `START` to DATE everywhere to avoid DuckDB VARCHAR comparisons  
✓ History list keeps temporal order via `list_aggregate(... ORDER BY START)`  
✓ Minor log clarifications & guard when zero rows returned  
"""

from __future__ import annotations
import argparse, datetime as dt, json, pathlib, sys
import duckdb
from typing import Optional

# --------------------------------------------------------------------- #
#  Project paths
# --------------------------------------------------------------------- #
ROOT    = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "duckdb" / "synthea.duckdb"
if not DB_PATH.exists():
    print(f"[prompt_gen] DuckDB not found at {DB_PATH}")
    sys.exit(1)

# --------------------------------------------------------------------- #
#  Helper: age + prompt composer
# --------------------------------------------------------------------- #
def build_prompt(rec: dict, cutoff_date: dt.date) -> str:
    """Compose natural-language prompt from patient record dictionary."""
    try:
        birth = dt.datetime.strptime(rec["birth"][:10], "%Y-%m-%d").date()
        age = int((cutoff_date - birth).days / 365.25)
    except Exception:
        age = "unknown-age"

    gender    = rec["gender"] or "unknown-gender"
    race      = (rec["race"] or "unknown-race").lower()
    ethnicity = (rec["ethnicity"] or "unknown-ethnicity").lower()

    # Use the detailed history string directly from the SQL query
    # Limit length if necessary, split by ', ' and rejoin with '; '
    hist_details_str = rec.get("past_dx_details", "")
    if hist_details_str:
        history_items = hist_details_str.split(', ') # Split based on SQL aggregation separator
        # Optionally limit the number of history items shown
        max_hist_items = 40
        hist_str = "; ".join(history_items[:max_hist_items])
    else:
        hist_str = "no previous diagnoses"

    return (
        f"{age}-year-old {gender} {race} {ethnicity} patient. "
        f"History of: {hist_str}."
    )

# --------------------------------------------------------------------- #
#  SQL template
# --------------------------------------------------------------------- #
SQL_TMPL = """
-- SQL Query Enhanced to include descriptions in history
WITH params AS (
    SELECT approx_quantile(CAST(START AS DATE), 0.8) AS cutoff
    FROM   conditions
), history AS (
    SELECT  c.PATIENT,
            -- Aggregate CODE and DESCRIPTION into an ordered list, then convert to string
            array_to_string(list(c.CODE || ' (' || c.DESCRIPTION || ')' ORDER BY CAST(c.START AS DATE)), ', ') AS past_dx_details,
            -- Keep the list of codes separately for the novelty filter if needed
            list(c.CODE ORDER BY CAST(c.START AS DATE)) AS past_dx_codes,
            any_value(p.GENDER)                                        AS gender,
            any_value(p.RACE)                                          AS race,
            any_value(p.ETHNICITY)                                     AS ethnicity,
            any_value(p.BIRTHDATE)                                     AS birth,
            any_value(params.cutoff)                                   AS cutoff
    FROM    conditions c
    CROSS JOIN params
    JOIN    patients p ON p.Id = c.PATIENT
    WHERE CAST(c.START AS DATE) < params.cutoff -- Apply time filter here
    GROUP BY c.PATIENT
), FirstFutureDx AS (
    -- Calculate rn and filter *inside* this CTE
    SELECT * FROM (
        SELECT  c.PATIENT,
                c.CODE AS future_dx,
                c.DESCRIPTION AS future_dx_desc, -- Also fetch description for future dx if needed
                ROW_NUMBER() OVER (
                    PARTITION BY c.PATIENT ORDER BY CAST(c.START AS DATE)
                ) AS rn
        FROM    conditions c
        JOIN    params ON CAST(c.START AS DATE) >= params.cutoff
    ) ranked
    WHERE rn = 1 -- Apply filter here
)
SELECT h.*, f.future_dx
FROM history h
JOIN FirstFutureDx f ON h.PATIENT = f.PATIENT
{novelty_filter} -- Placeholder for conditional WHERE clause
-- LIMIT clause will be added dynamically by _build_sql if needed
"""

# --------------------------------------------------------------------- #
#  Streaming generator
# --------------------------------------------------------------------- #

# Helper function to construct the final SQL query
def _build_sql(allow_repeats: bool = False, limit: Optional[int] = None):
    if allow_repeats:
        novelty_clause = "" # No filter needed
    else:
        # Modify the novelty filter to use past_dx_codes
        novelty_clause = """
            WHERE NOT EXISTS (
                  SELECT 1
                  FROM UNNEST(h.past_dx_codes) AS x(code) -- Use the code-only list
                  WHERE CAST(x.code AS VARCHAR) = CAST(f.future_dx AS VARCHAR)
            )
        """
    base_sql = SQL_TMPL.format(novelty_filter=novelty_clause)
    # Append LIMIT clause if provided and positive
    if limit is not None and limit > 0:
        # IMPORTANT: Ensure there isn't already a semicolon if base_sql ends with one
        base_sql = base_sql.rstrip().rstrip(';')
        return f"{base_sql}\nLIMIT {limit};"
    else:
        return base_sql

def stream_rows(allow_repeats: bool = False, batch: int = 1000, limit: Optional[int] = None):
    """Yield dicts of {'text': prompt, 'label': code} directly from DuckDB."""
    sql = _build_sql(allow_repeats, limit)

    with duckdb.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute(sql)

        cols = [d[0] for d in cur.description]
        rows_seen = 0

        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                if rows_seen == 0:
                    print("[prompt_gen] Query returned zero rows!")
                break

            for tup in rows:
                rows_seen += 1
                rec = dict(zip(cols, tup))
                prompt = build_prompt(rec, rec["cutoff"])
                label  = str(rec["future_dx"])

                yield {
                    "text":  f"{prompt}\n\nAI: {label}",
                    "label": label
                }

# Function to get the total count of rows the query will produce
def count_rows(allow_repeats: bool = False) -> int:
    """Connects to DuckDB and runs a COUNT(*) query on the base query (pre-limit)."""
    # Build SQL *without* the limit for counting total potential rows
    base_sql = _build_sql(allow_repeats, limit=None)
    count_sql = f"SELECT COUNT(*) FROM ({base_sql.rstrip().rstrip(';')}) AS count_subquery;"

    try:
        with duckdb.connect(DB_PATH) as con:
            result = con.execute(count_sql).fetchone()
            return result[0] if result else 0
    except Exception as e:
        print(f"[prompt_gen] Error counting rows: {e}")
        return 0 # Return 0 on error

# --------------------------------------------------------------------- #
#  CLI entry
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Stream prompt/label pairs")
    parser.add_argument("--allow_repeats", action="store_true",
                        help="Keep rows where label appears in history.")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="DuckDB fetchmany batch size.")
    parser.add_argument("--preview", type=int, default=0,
                        help="Print N samples then exit (0 = stream indefinitely).")
    args = parser.parse_args()

    count = 0
    for sample in stream_rows(args.allow_repeats, args.batch_size):
        if args.preview and count >= args.preview:
            break
        print(json.dumps(sample, ensure_ascii=False))
        count += 1

    if args.preview:
        print(f"\nPreviewed {count} sample(s).")

if __name__ == "__main__":
    main()
