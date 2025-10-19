"""
fx_processing_pipeline.py
-------------------------
Self-contained FX pipeline with CSV support.

Features
- read_tabular(): auto-detect CSV (falls back to Excel if provided)
- load(): accepts path or DataFrame; builds 'mid' if only bid/ask present
- validate FX schema lightly via construction and sorting (no hard stop)
- resampling/interpolation helpers (resample optional via config; here we only interpolate)
- write_csv(): save processed output to CSV
- CLI: process & save with one command
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

# Only CSV is officially supported here; Excel is attempted as a fallback on read.
CSV_EXTS = {".csv"}

def read_tabular(path: Union[str, Path], **io_kwargs) -> pd.DataFrame:
    """
    Load a tabular file.

    Priority:
      1) If extension is CSV/empty → read_csv
      2) Otherwise try read_excel, and if that fails → read_csv

    Notes:
      - This lets you pass files without an extension (common in temp exports)
      - Additional pandas read_* kwargs can be passed in **io_kwargs
    """
    p = Path(path)
    ext = p.suffix.lower()

    # Primary path: CSV (or no extension assumed CSV)
    if ext in CSV_EXTS or ext == "":
        return pd.read_csv(p, **io_kwargs)

    # Fallback path: try Excel, then fall back to CSV if Excel fails
    try:
        return pd.read_excel(p, **io_kwargs)
    except Exception:
        return pd.read_csv(p, **io_kwargs)

def write_csv(
    df: pd.DataFrame,
    path: Union[str, Path],
    index: bool = True,
    float_format: Optional[str] = None,
    **io_kwargs
) -> None:
    """
    Write a DataFrame to CSV.

    Args
    ----
    df : DataFrame (ideally with a timezone-aware DatetimeIndex in UTC).
    path : Output CSV path.
    index : Whether to write index to file (True keeps the timestamp column in CSV).
    float_format : Numeric format for floats, e.g., '%.6f'.
    **io_kwargs : Passed through to pandas.DataFrame.to_csv().
    """
    p = Path(path)
    # Ensure output directory exists
    p.parent.mkdir(parents=True, exist_ok=True)

    # Keep a copy for safe manipulation before writing
    if isinstance(df.index, pd.DatetimeIndex):
        # (Optional) If you prefer explicit ISO strings in CSV, uncomment below
        # df_to_write = df.copy()
        # df_to_write.index = df_to_write.index.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S%z")
        # However, pandas already writes tz-aware datetimes in a readable way.
        df_to_write = df
    else:
        df_to_write = df

    # ⬇️ Ensure strictly ascending timestamp/index before writing.
    # Using stable mergesort avoids reordering equal timestamps unpredictably.
    if isinstance(df_to_write.index, pd.DatetimeIndex):
        df_to_write = df_to_write.sort_index(ascending=True, kind="mergesort").copy()

    # Finally write CSV
    df_to_write.to_csv(p, index=index, float_format=float_format, **io_kwargs)

# Common alias mappings so various column header styles still work
ALIAS_MAP_FX = {
    "time": "timestamp",
    "datetime": "timestamp",
    "date": "timestamp",
    "symbol": "pair",
    "pair": "pair",
    "price": "mid",
}

def normalize_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to canonical FX column names using ALIAS_MAP_FX.

    Examples:
      - 'time', 'datetime', 'date'  → 'timestamp'
      - 'symbol'                    → 'pair'
      - 'price'                     → 'mid'
    """
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for alias, canon in ALIAS_MAP_FX.items():
        if alias in lower:
            rename[lower[alias]] = canon
    return df.rename(columns=rename)

def ensure_datetime_index(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Ensure the DataFrame has a tz-aware DatetimeIndex (UTC), sorted ascending.

    Behavior:
      - If 'timestamp' col exists → parse, drop NA, sort, set as index
      - Else try known time-like columns ('date','datetime','time')
      - If index already DatetimeIndex → just normalize to UTC
      - Always localize to UTC (or convert to UTC if tz-aware)
    """
    df = df.copy()

    # 1) Primary 'timestamp' column
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=False)
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).set_index(ts_col)

    # 2) Try other common time-like columns
    elif not isinstance(df.index, pd.DatetimeIndex):
        for cand in ("date", "datetime", "time"):
            if cand in df.columns:
                df[cand] = pd.to_datetime(df[cand], errors="coerce", utc=False)
                df = df.dropna(subset=[cand]).sort_values(cand).set_index(cand)
                df.index.name = ts_col  # rename index to 'timestamp'
                break

    # 3) Normalize timezone to UTC
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    return df

def resample_if_needed(df: pd.DataFrame, freq: Optional[str]) -> pd.DataFrame:
    """
    Optional resampling helper.

    - If freq is None/Falsey → return df unchanged
    - Otherwise resample to the given frequency using 'last' for all columns
      and drop rows where all values are NA.
    """
    if not freq:
        return df
    agg = {c: "last" for c in df.columns}
    return df.resample(freq).agg(agg).dropna(how="all")

def interpolate_missing(df: pd.DataFrame, method: str = "time") -> pd.DataFrame:
    """
    Gap-filling helper for regularly/irregularly spaced time series.

    Supported:
      - 'ffill'  : forward-fill
      - 'time'   : time-based interpolation (good for DatetimeIndex)
      - 'linear' : linear interpolation on numeric index order
    """
    if method == "ffill":
        return df.ffill()
    if method in ("time", "linear"):
        return df.interpolate(method=method, limit_area="inside")
    raise ValueError("Unknown method: choose from {'time','linear','ffill'}")

@dataclass
class FXConfig:
    """
    Configuration for pipeline behavior.

    output_freq         : Optional resampling frequency (e.g. '1D', '15min').
                          (Note: this script’s CLI does not expose it; kept for parity.)
    interpolate_method  : Interpolation strategy for gaps.
    allowed_pairs       : Optional whitelist filter for 'pair' column, e.g. ('EURUSD','USDEUR').
    """
    output_freq: Optional[str] = None
    interpolate_method: str = "time"
    allowed_pairs: Optional[Tuple[str, ...]] = None  # e.g., ("EURUSD","USDEUR")

class FXPipeline:
    """
    End-to-end FX processing pipeline:
      - Load (path or DataFrame)
      - Normalize column names
      - Force UTC DatetimeIndex (sorted)
      - Build 'mid' from 'bid'/'ask' if needed
      - Optional pair filter
      - Optional resampling + interpolation
    """
    def __init__(self, config: FXConfig = FXConfig()):
        self.cfg = config

    def load(self, df_or_path: Union[str, Path, pd.DataFrame], **io_kwargs) -> pd.DataFrame:
        """
        Load and normalize raw data.

        Steps:
          1) Read (if path) or copy (if DataFrame)
          2) Normalize column aliases
          3) Ensure DatetimeIndex in UTC (sorted)
          4) Create 'mid' if both 'bid' and 'ask' exist but 'mid' does not
          5) Apply allowed_pairs filter if configured
        """
        # Read or copy input
        if isinstance(df_or_path, (str, Path)):
            df = read_tabular(df_or_path, **io_kwargs)
        else:
            df = df_or_path.copy()

        # Normalize schema and index
        df = normalize_column_aliases(df)
        df = ensure_datetime_index(df)

        # Build 'mid' if only bid/ask provided
        if "mid" not in df.columns and {"bid", "ask"}.issubset(df.columns):
            df["mid"] = (df["bid"].astype(float) + df["ask"].astype(float)) / 2.0

        # Optional pair filtering
        if self.cfg.allowed_pairs:
            df = df[df["pair"].astype(str).isin(self.cfg.allowed_pairs)]

        return df

    def align_and_resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample to target frequency if configured (no-op in this CLI by default).
        """
        return resample_if_needed(df, self.cfg.output_freq)

    def interpolate_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the configured interpolation method to fill internal gaps.
        """
        return interpolate_missing(df, self.cfg.interpolate_method)

    def process(self, df_or_path: Union[str, Path, pd.DataFrame], **io_kwargs) -> pd.DataFrame:
        """
        Load → (optional) resample → (optional) interpolate; return processed DataFrame.
        """
        df = self.load(df_or_path, **io_kwargs)
        df = self.align_and_resample(df)
        df = self.interpolate_missing(df)
        return df

# -------------------- CLI / Script entrypoint --------------------
def parse_pairs(pairs: Optional[str]) -> Optional[Tuple[str, ...]]:
    """
    Parse a comma/space separated string of pairs into a tuple.

    Example inputs:
      - "EURUSD USDEUR"
      - "EURUSD,USDEUR"
    """
    if not pairs:
        return None
    parts = [p.strip().upper() for p in pairs.replace(",", " ").split() if p.strip()]
    return tuple(parts) if parts else None

def main():
    """
    Command-line entry point:
      - Reads an input CSV (or Excel fallback)
      - Interpolates gaps (method selectable)
      - Writes a processed CSV to the output path
    """
    ap = argparse.ArgumentParser(description="Process FX time series and save the result.")
    ap.add_argument(
        "-i", "--input",
        required=False,
        default="input_data/USD_EUR_daily_full.csv",
        help="Input CSV path (Excel is attempted if CSV fails)."
    )
    ap.add_argument(
        "-o", "--output",
        required=False,
        default="processing_output/fx_processed.csv",
        help="Output CSV path."
    )
    ap.add_argument(
        "--method",
        default="time",
        choices=["time", "linear", "ffill"],
        help="Interpolation method for gaps."
    )
    ap.add_argument(
        "--float-format",
        default=None,
        help="Optional numeric format for CSV (e.g., '%.6f')."
    )
    args = ap.parse_args()

    # Configure pipeline (resampling not exposed here; interpolation is)
    cfg = FXConfig(interpolate_method=args.method)
    pipe = FXPipeline(cfg)

    # Read/process with optional pandas read_* kwargs via io_kwargs if needed
    io_kwargs = {}

    # Execute processing
    df = pipe.process(args.input, **io_kwargs)

    # Save result
    write_csv(
        df=df,
        path=args.output,
        float_format=args.float_format,
        index=True,
    )
    print(f"[SAVE] Wrote processed FX to: {args.output}  rows={len(df):,}")

if __name__ == "__main__":
    # Example direct run:
    # python fx_processing_pipeline.py -i input_data/USD_EUR_daily_full.csv -o processing_output/fx_processed.csv
    main()
