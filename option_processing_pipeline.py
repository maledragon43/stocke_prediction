"""
----------------------------------------
Self-contained options pipeline with CSV support and a CLI.

What this module does
- read_csv(): auto-detect basic CSV (with a fallback that also tolerates a wrong extension)
- write_csv(): save processed output to CSV
- normalize_column_aliases(): map common column aliases to canonical names
- ensure_datetime_index(): ensure a UTC DatetimeIndex based on a timestamp-like column
- validate(): lightweight schema checks + helpful prints
- OptionPipeline: load → snapshot_at → scope_by_expiry → nearest_by_delta → process
- CLI: run end-to-end and write results

Notes
- Only CSV is considered for writes; reads try CSV first and gracefully fall back to Excel.
- "expiry" is expected to be parseable to datetime; we'll compute DTE relative to the requested `asof`.
- `delta` should be provided in the input if you want `nearest_by_delta()` to work (it ranks by |abs(delta)−target|).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

# ------------------------------
# I/O helpers (CSV-centric)
# ------------------------------
# Accept only .csv for the saving path, but allow empty suffix (we'll append .csv)
CSV_EXTS   = {".csv"}

def read_csv(path: Union[str, Path], **io_kwargs) -> pd.DataFrame:
    """Read a tabular file into a DataFrame.

    Behavior
    - If the file extension is .csv or empty → read via pandas.read_csv.
    - Otherwise, try read_excel (to be forgiving if the user gave .xlsx by mistake),
      and if that fails, fall back to read_csv again.

    Parameters
    ----------
    path : str | Path
        Path to a CSV (preferred) or other delimited file; Excel is tolerated.
    **io_kwargs : any
        Extra keyword args forwarded to pandas readers (e.g., sep=",", encoding=...).

    Returns
    -------
    pd.DataFrame
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in CSV_EXTS or ext == "":
        return pd.read_csv(p, **io_kwargs)
    try:
        # Be forgiving: if user passed an Excel file, try to read it
        return pd.read_excel(p, **io_kwargs)
    except Exception:
        # Final fallback: try CSV again
        return pd.read_csv(p, **io_kwargs)


def write_csv(df: pd.DataFrame, path: Union[str, Path], **io_kwargs) -> None:
    """Save a DataFrame to CSV, picking sensible defaults.

    Behavior
    - If path has .csv (or no suffix), write CSV with index=False by default.
    - If path has an unknown/unsupported suffix, we still write a .csv beside it.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write.
    path : str | Path
        Desired output path; if suffix is missing, ".csv" is appended.
    **io_kwargs : any
        Forwarded to DataFrame.to_csv (e.g., sep=",", encoding=...).
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in CSV_EXTS or ext == "":
        # default to index=False unless the caller overrides
        if "index" not in io_kwargs:
            io_kwargs["index"] = False
        df.to_csv(p if ext else p.with_suffix(".csv"), **io_kwargs)

    # If extension wasn't .csv, also emit a CSV as a safe fallback
    if "index" not in io_kwargs:
        io_kwargs["index"] = False
    df.to_csv(p.with_suffix(".csv"), **io_kwargs)


# -------------------------------------------------
# Column alias normalization (robust to variations)
# -------------------------------------------------
ALIAS_MAP_OPT = {
    # timestamps
    "time": "timestamp",
    "datetime": "timestamp",
    "date": "timestamp",

    # underlying identifiers
    "underlying": "symbol",
    "ticker": "symbol",
    "symbol": "symbol",

    # expiries
    "expiration": "expiry",
    "expiry": "expiry",
    "exp": "expiry",

    # option type / CP flag
    "type": "option_type",
    "cp": "option_type",
    "optiontype": "option_type",

    # prices / greeks / misc
    "bid": "bid",
    "ask": "ask",
    "mid": "mid",
    "delta": "delta",
    "curr": "currency",
    "currency": "currency",

    # strikes
    "strikeprice": "strike",
}

def normalize_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to a canonical set using ALIAS_MAP_OPT.

    Examples:
    - "expiration" → "expiry"
    - "type" → "option_type"
    - "datetime" → "timestamp"
    """
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for alias, canon in ALIAS_MAP_OPT.items():
        if alias in lower:
            rename[lower[alias]] = canon
    return df.rename(columns=rename)


# ----------------------------------------------
# Datetime index utilities (force UTC index)
# ----------------------------------------------

def ensure_datetime_index(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Ensure `df` has a UTC DatetimeIndex.

    Rules
    - If `ts_col` exists: parse to datetime, sort, and set as index.
    - Else, look for common alternatives ("date", "datetime", "time") and use the first found.
    - If the resulting index has no tz → localize to UTC; otherwise convert to UTC.
    """
    df = df.copy()

    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=False)
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).set_index(ts_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try common names if the canonical column isn't present
        for cand in ("date", "datetime", "time"):
            if cand in df.columns:
                df[cand] = pd.to_datetime(df[cand], errors="coerce", utc=False)
                df = df.dropna(subset=[cand]).sort_values(cand).set_index(cand)
                df.index.name = ts_col
                break

    # Enforce UTC timezone for consistent time arithmetic
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


# ---------------------
# Basic schema checks
# ---------------------

def validate(df: pd.DataFrame) -> None:
    """Print friendly messages about the core schema and ordering.

    Required columns: symbol, expiry, strike, option_type, currency
    Additional expectations:
    - Either `mid` exists, or both `bid` and `ask` exist (we compute mid if needed)
    - `delta` is required if you plan to call nearest_by_delta
    - Index should be time-sorted (monotonic increasing)
    """
    req = ["symbol", "expiry", "strike", "option_type", "currency"]
    issues = []

    miss = [c for c in req if c not in df.columns]
    if miss:
        issues.append(f"Missing: {miss}")

    if "mid" not in df.columns and {"bid", "ask"}.issubset(df.columns):
        # fine, we'll compute mid in load()
        pass
    elif "mid" not in df.columns:
        issues.append("Need 'mid' or both 'bid' and 'ask'")

    if "delta" not in df.columns:
        issues.append("Missing 'delta' (needed for nearest_by_delta)")

    if df.empty:
        issues.append("Empty DataFrame")

    if not df.index.is_monotonic_increasing:
        issues.append("Index not sorted")

    if issues:
        print("[VALIDATION] Options:\n  - " + "\n  - ".join(issues))
    else:
        print("[VALIDATION] Options: OK")


# -----------------------------
# Configuration dataclass
# -----------------------------
@dataclass
class OptionConfig:
    """User-tunable targets used during selection."""
    # e.g., select near 10Δ, 25Δ, 50Δ options
    target_deltas: Tuple[float, ...] = (0.10, 0.25, 0.50)
    # e.g., select expiries ~ 1w, 1m, 3m (by days-to-expiry)
    target_expiries_days: Tuple[int, ...] = (7, 30, 90)


# -----------------------------
# Core pipeline class
# -----------------------------
class OptionPipeline:
    def __init__(self, config: OptionConfig = OptionConfig()):
        self.cfg = config

    # -----------
    # Load stage
    # -----------
    def load(self, df_or_path: Union[str, Path, pd.DataFrame], **io_kwargs) -> pd.DataFrame:
        """Load, normalize, time-index, and compute mid if needed.

        Steps
        1) Read → DataFrame
        2) Normalize column names to canonical set
        3) Ensure UTC DatetimeIndex (sorted)
        4) Compute `mid` = (bid+ask)/2 if `mid` missing but bid/ask present
        5) Normalize option_type to {"C","P"}
        6) Run a quick validation printout
        """
        if isinstance(df_or_path, (str, Path)):
            df = read_csv(df_or_path, **io_kwargs)
        else:
            df = df_or_path.copy()

        df = normalize_column_aliases(df)
        df = ensure_datetime_index(df)

        # Create mid if we have both bid and ask
        if "mid" not in df.columns and {"bid", "ask"}.issubset(df.columns):
            df["mid"] = (df["ask"].astype(float) + df["bid"].astype(float)) / 2.0

        # Normalize calls/puts regardless of input casing/value
        if "option_type" in df.columns:
            df["option_type"] = (
                df["option_type"].astype(str).str.upper().map({
                    "C": "C", "CALL": "C", "P": "P", "PUT": "P"
                }).fillna(df["option_type"])
            )

        validate(df)
        return df

    # ----------------------------
    # Time-aware snapshot utility
    # ----------------------------
    def snapshot_at(self, chain: pd.DataFrame, asof: Union[str, pd.Timestamp]) -> pd.DataFrame:
        """Pick the latest row *per contract* at or before `asof`.

        Grouping key (a "contract") is (symbol, expiry, strike, option_type).
        We filter rows whose timestamp ≤ `asof`, then take the last row for each group.
        """
        asof_ts = pd.Timestamp(asof)
        asof_ts = asof_ts.tz_localize("UTC") if asof_ts.tzinfo is None else asof_ts.tz_convert("UTC")

        # Keep only rows up to asof
        x = chain.loc[chain.index <= asof_ts].copy()
        if x.empty:
            return x

        x = x.sort_index()
        group_cols = ["symbol", "expiry", "strike", "option_type"]
        for col in group_cols:
            if col not in x.columns:
                raise ValueError(f"snapshot_at: missing required column '{col}'")

        # Take the last row per group (most recent at/before asof)
        snap = x.groupby(group_cols, sort=False).tail(1)

        # Bring timestamp back as a normal column for the downstream sort
        return snap.reset_index().rename(columns={"index": "timestamp"})

    # ----------------------------------
    # Expiry scoping around target DTEs
    # ----------------------------------
    def scope_by_expiry(self, chain: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
        """From a snapshot, choose expiries nearest to configured DTE buckets.

        For each desired day bucket (e.g., 7/30/90), compute |DTE−bucket| and
        pick the expiry date that minimizes this distance (ahead of `asof`).
        """
        asof = (
            pd.Timestamp(asof).tz_localize("UTC")
            if pd.Timestamp(asof).tzinfo is None
            else pd.Timestamp(asof).tz_convert("UTC")
        )
        x = chain.copy()

        # Ensure expiry is timezone-aware for consistent subtraction
        x["expiry"] = pd.to_datetime(x["expiry"], errors="coerce", utc=True)
        x["dte"] = (x["expiry"] - asof).dt.days  # integer days to expiry

        selected: List[pd.DataFrame] = []
        for days in self.cfg.target_expiries_days:
            # Only non-negative DTE (avoid already-expired)
            subset = x[x["dte"] >= 0].assign(exp_dist=lambda s: (s["dte"] - days).abs())
            if subset.empty:
                continue
            # Find the expiry date closest to the target bucket
            exp_date = subset.sort_values(["exp_dist"]).head(1)["expiry"].iloc[0]
            selected.append(x[x["expiry"] == exp_date])

        # If nothing matched (e.g., all expired), return empty frame of same columns
        return pd.concat(selected, axis=0) if selected else x.iloc[0:0]

    # ----------------------------------
    # Delta proximity picker
    # ----------------------------------
    def nearest_by_delta(self, chain: pd.DataFrame) -> pd.DataFrame:
        """Pick one contract per target delta using a simple distance metric.

        Metric
        - Rank by |abs(delta) − target| ascending.
        - If bid/ask are available, we tiebreak by tightest spread = |ask−bid|.
        """
        if "delta" not in chain.columns:
            raise ValueError("Missing 'delta'")

        picks: List[pd.DataFrame] = []
        for d in self.cfg.target_deltas:
            tmp = chain.assign(delta_dist=(chain["delta"].abs() - d).abs())
            if {"bid", "ask"}.issubset(tmp.columns):
                tmp["spread"] = (tmp["ask"] - tmp["bid"]).abs()
                pick = tmp.sort_values(["delta_dist", "spread"]).head(1)
            else:
                pick = tmp.sort_values(["delta_dist"]).head(1)

            pick = pick.copy()
            pick["target_delta"] = d
            picks.append(pick)

        return pd.concat(picks, axis=0)

    # ------------------------------
    # End-to-end processing routine
    # ------------------------------
    def process(self, chain: pd.DataFrame, asof: Optional[Union[str, pd.Timestamp]] = None) -> pd.DataFrame:
        """Run the full selection flow and return a tidy DataFrame.

        Steps
        1) If `asof` not provided, use the last timestamp in the data.
        2) snapshot_at: latest row per contract at/before asof
        3) scope_by_expiry: choose expiries near configured DTE buckets
        4) nearest_by_delta: pick one contract per target delta per (symbol, option_type, expiry)
        5) Order columns for readability
        """
        # 1) Establish the reference timestamp
        if asof is None:
            if isinstance(chain.index, pd.DatetimeIndex) and len(chain.index) > 0:
                asof = chain.index.max()
            else:
                raise ValueError("process: cannot infer 'asof' from data; please pass asof")

        # 2) Latest row per contract as of `asof`
        snap = self.snapshot_at(chain, asof)
        if snap.empty:
            return snap

        # 3) Scope by nearest target expiries
        scoped = self.scope_by_expiry(snap, asof)

        # 4) Within each (symbol, type, expiry), pick nearest-by-delta
        results: List[pd.DataFrame] = []
        for (sym, opt_type, exp), sub in scoped.groupby(["symbol", "option_type", "expiry"]):
            try:
                picks = self.nearest_by_delta(sub)
                picks = picks.assign(symbol=sym, option_type=opt_type, expiry=exp)
                results.append(picks)
            except Exception:
                # e.g., missing delta for that subgroup → skip
                continue

        out = pd.concat(results, axis=0) if results else scoped.iloc[0:0]

        # 5) Pretty column ordering
        preferred = [
            "timestamp", "symbol", "expiry", "dte", "option_type", "strike",
            "bid", "ask", "mid", "delta", "target_delta", "currency"
        ]
        cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]

        # Ensure we have a visible timestamp column for the final CSV
        if "timestamp" in out.columns:
            out = out[cols].sort_values(["symbol", "expiry", "option_type", "target_delta", "strike"])
        else:
            out = out.reset_index().rename(columns={"index": "timestamp"})
            out = out[cols].sort_values(["symbol", "expiry", "option_type", "target_delta", "strike"])

        return out


# ---------------- CLI wiring ---------------- #

def _parse_args() -> argparse.Namespace:
    """Define and parse command-line options for the pipeline.

    directory layout matches the expected `input_data/` and `processing_output/`.
    """
    p = argparse.ArgumentParser(description="Process options chain and save selected contracts.")
    p.add_argument("-i", "--input", required=False, default="input_data/options.csv", help="Input CSV file path")
    p.add_argument("-o", "--output", required=False, default="processing_output/option_processed.csv", help="Output CSV file path")
    p.add_argument("--asof", help="Timestamp for snapshot (e.g., '2025-10-06 16:00:00Z')")
    p.add_argument("--csv-sep", default=",", help="CSV delimiter when saving .csv (default ',')")
    p.add_argument("--target-deltas", default="0.10,0.25,0.50", help="Comma list of target deltas")
    p.add_argument("--target-expiries", default="7,30,90", help="Comma list of target DTE days")
    return p.parse_args()


def _parse_tuple_floats(s: str) -> Tuple[float, ...]:
    """Turn a comma-separated string into a tuple of floats (e.g., "0.1,0.25")."""
    return tuple(float(x.strip()) for x in s.split(",") if str(x).strip() != "")


def _parse_tuple_ints(s: str) -> Tuple[int, ...]:
    """Turn a comma-separated string into a tuple of ints (e.g., "7,30,90")."""
    return tuple(int(float(x.strip())) for x in s.split(",") if str(x).strip() != "")


def main():
    try:
        # Parse CLI flags
        args = _parse_args()

        # Build config from flags
        cfg = OptionConfig(
            target_deltas=_parse_tuple_floats(args.target_deltas),
            target_expiries_days=_parse_tuple_ints(args.target_expiries),
        )

        # Instantiate the pipeline
        pipe = OptionPipeline(cfg)

        # Load the chain (CSV by default)
        df = pipe.load(args.input)

        # If --asof omitted, use the last available timestamp
        asof = args.asof or (df.index.max().isoformat() if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0 else None)
        if asof is None:
            raise SystemExit("Could not infer --asof; please supply it explicitly.")

        # Run the selection logic
        out = pipe.process(df, asof=asof)

        # Prepare save kwargs for CSV
        save_kwargs = {}
        if args.output.lower().endswith((".csv")):
            save_kwargs["sep"] = args.csv_sep

        # Write to disk
        write_csv(out, args.output, **save_kwargs)
        print(f"[OK] Saved {len(out):,} rows → {args.output}")

    except Exception as e:
        # Print a concise error line for easy debugging in logs/CI
        print("[ERROR]", e)
        
if __name__ == "__main__":
    # Example direct run:
    # python option_processing_pipeline.py -i input_data/options.csv -o processing_output/option_processed.csv
    main()