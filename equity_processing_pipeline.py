
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import argparse

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
# Recognized file extensions. Anything else will try Excel first, then CSV.
CSV_EXTS   = {".csv"}

def read_tabular(path: Union[str, Path], **io_kwargs) -> pd.DataFrame:
    """
    Read a delimited (CSV) or into a DataFrame.

    Parameters
    ----------
    path : str | Path
        File path to read.
    Returns
    -------
    DataFrame
        Raw DataFrame as read—no column normalization or datetime processing yet.

    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in CSV_EXTS or ext == "":
        return pd.read_csv(p, **io_kwargs)
    # Unknown extension: best‑effort fallback
    try:
        return pd.read_excel(p, **io_kwargs)
    except Exception:
        return pd.read_csv(p, **io_kwargs)

# Aliases frequently found in vendor files mapped to canonical names.
ALIAS_MAP_EQUITY: Dict[str, str] = {
    "time": "timestamp",
    "datetime": "timestamp",
    "date": "timestamp",
    "ticker": "symbol",
    "sym": "symbol",
    "curr": "currency",
    # compact OHLCV aliases used by some APIs
    "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
}

def normalize_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename common variant column names to a canonical schema.
    This reduces the need to pre-clean vendor files.

    
    --------
    - 'date' or 'datetime'  -> 'timestamp'
    - 'ticker', 'sym'       -> 'symbol'
    - 'curr'                -> 'currency'
    - 'o','h','l','c','v'   -> 'open','high','low','close','volume'
    """
    lower = {c.lower(): c for c in df.columns}  # map lower->original
    rename: Dict[str, str] = {}
    for alias, canon in ALIAS_MAP_EQUITY.items():
        if alias in lower:
            rename[lower[alias]] = canon
    return df.rename(columns=rename)

# ---------------------------------------------------------------------------
# Time & resampling
# ---------------------------------------------------------------------------
def ensure_datetime_index(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Ensure the DataFrame is indexed by a datetime index called 'timestamp'.

    Strategy
    --------
    1) If 'timestamp' exists → parse to datetime and set as index.
    2) Else if index is already DatetimeIndex → keep it (no parsing).
    3) Else try fallback columns in order: 'date' → 'datetime' → 'time'.

    Notes
    -----
    - Parsing uses errors='coerce' so unparsable rows become NaT and are dropped.
    - No timezone is applied here; we only parse to naive datetimes.
      Timezone handling is performed by `localize_to_timezone` later.
    """
    df = df.copy()
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=False)
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).set_index(ts_col)
    elif isinstance(df.index, pd.DatetimeIndex):
        # Already datetime‑indexed—just ensure sorted and NaT‑free.
        df = df[~df.index.to_series().isna()].sort_index()
    else:
        for cand in ("date", "datetime", "time"):
            if cand in df.columns:
                df[cand] = pd.to_datetime(df[cand], errors="coerce", utc=False)
                df = df.dropna(subset=[cand]).sort_values(cand).set_index(cand)
                df.index.name = ts_col
                break
    return df

def resample_if_needed(df: pd.DataFrame, freq: Optional[str]) -> pd.DataFrame:
    """
    Resample to a coarser time frequency using OHLCV‑aware aggregations.

    Aggregations
    ------------
    - open  : first
    - high  : max
    - low   : min
    - close : last
    - volume: sum
    - any other column: last

    Parameters
    ----------
    df : DataFrame
        Time‑indexed data.
    freq : str | None
        Pandas offset alias (e.g., '1D', '1H'). If None, return as‑is.
    """
    if not freq:
        return df
    agg = {c: "last" for c in df.columns}
    for c in df.columns:
        cl = c.lower()
        if cl == "open":   agg[c] = "first"
        elif cl == "high": agg[c] = "max"
        elif cl == "low":  agg[c] = "min"
        elif cl == "volume": agg[c] = "sum"
    return df.resample(freq).agg(agg).dropna(how="all")

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame, required: List[str], name: str = "Equity") -> None:
    """
    Print a simple validation report listing missing columns, NaNs, etc.

    Checks
    ------
    - all required columns exist
    - frame is non‑empty
    - index is sorted ascending
    - columns with any NaN values
    """
    issues: List[str] = []
    miss = [c for c in required if c not in df.columns]
    if miss: issues.append(f"Missing columns: {miss}")
    if df.empty: issues.append("DataFrame is empty")
    if not df.index.is_monotonic_increasing: issues.append("Index not sorted")
    nan_cols = [c for c in df.columns if df[c].isna().any()]
    if nan_cols: issues.append(f"NaNs in: {nan_cols}")
    if issues:
        print(f"[VALIDATION] {name}:\n  - " + "\n  - ".join(issues))
    else:
        print(f"[VALIDATION] {name}: OK")

# ---------------------------------------------------------------------------
# Interpolation & Outliers
# ---------------------------------------------------------------------------
def interpolate_missing(df: pd.DataFrame, method: str = "time",
                        max_gap: Optional[pd.Timedelta] = None) -> pd.DataFrame:
    """
    Fill internal gaps to produce a continuous series suitable for modeling.

    Parameters
    ----------
    method : {'time','linear','ffill'}
        'time'/'linear' use pandas.interpolate(limit_area='inside').
        'ffill' forward‑fills previous value.
    max_gap : Timedelta | None
        If provided, values after a gap larger than max_gap are NOT filled:
        - For 'ffill': rows after large gaps are set to NaN before ffill.
        - For 'time'/'linear': interpolated values beyond the mask are nulled.

    Returns
    -------
    DataFrame
        Data with only reasonable fills applied.
    """
    x = df.copy()
    if method == "ffill":
        if max_gap is not None:
            gaps = x.index.to_series().diff() > max_gap
            x.loc[gaps.values, :] = np.nan
        return x.ffill()
    if method in ("time", "linear"):
        interp = x.interpolate(method=method, limit_area="inside")
        if max_gap is not None:
            mask = x.index.to_series().diff() <= max_gap
            interp[~mask.values] = np.nan
        return interp
    raise ValueError(f"Unknown interpolation method: {method}")

def add_rolling_zscore(df: pd.DataFrame, col: str, win: int = 50) -> pd.DataFrame:
    """
    Add a rolling z‑score column '<col>_z' using window 'win'.
    Useful to visualize price deviations before flagging outliers.
    """
    x = df.copy()
    mean = x[col].rolling(win, min_periods=max(5, win//5)).mean()
    std  = x[col].rolling(win, min_periods=max(5, win//5)).std(ddof=0)
    x[f"{col}_z"] = (x[col] - mean) / std
    return x

def detect_outliers(df: pd.DataFrame, price_col: str = "close", z_thresh: float = 4.0,
                    iforest_contamination: float = 0.01, random_state: int = 42) -> pd.DataFrame:
    """
    Flag outliers using two complementary methods:

    1) Rolling z‑score on 'price_col' (default: |z| > 4)
    2) IsolationForest on [price_col, returns] (when sufficient data)

    Notes
    -----
    - IsolationForest is trained only on non‑NaN rows to avoid spurious flags.
    - Output columns:
        * '<price_col>_z'   : rolling z‑score
        * 'outlier_z'       : boolean mask from z‑score threshold
        * 'ret'             : pct_change of price_col
        * 'outlier_iforest' : boolean mask from IsolationForest
    """
    x = df.copy()
    if price_col in x.columns:
        x = add_rolling_zscore(x, price_col)
        x["outlier_z"] = x[f"{price_col}_z"].abs() > z_thresh
        x["ret"] = x[price_col].pct_change()
    cols = [c for c in ["ret", price_col] if c in x.columns]
    if cols:
        clean = x[cols].dropna()
        if not clean.empty:
            model = IsolationForest(contamination=iforest_contamination, random_state=random_state)
            model.fit(clean)
            preds = pd.Series(model.predict(clean), index=clean.index).eq(-1)
            x["outlier_iforest"] = False
            x.loc[preds.index, "outlier_iforest"] = preds
        else:
            x["outlier_iforest"] = False
    return x

# ---------------------------------------------------------------------------
# Pipeline object
# ---------------------------------------------------------------------------
@dataclass
class EquityConfig:
    """
    Tunables for the pipeline.

    output_freq : resampling frequency, e.g. '1D'. If None, keep original freq.
    interpolate_method : 'time' (default), 'linear', or 'ffill'.
    max_gap : do NOT fill across gaps greater than this duration.
    zscore_window / zscore_threshold : for rolling z‑score outlier flags.
    iforest_contamination / iforest_random_state : IsolationForest controls.
    """
    output_tz: str = "UTC"
    output_freq: Optional[str] = None
    interpolate_method: str = "time"
    max_gap: Optional[pd.Timedelta] = None
    zscore_window: int = 50
    zscore_threshold: float = 4.0
    iforest_contamination: float = 0.01
    iforest_random_state: int = 42

class EquityPipeline:
    """
    High‑level orchestrator for loading, standardizing, and exporting equities.
    """
    def __init__(self, config: EquityConfig = EquityConfig()):
        self.cfg = config

    def load(self, df_or_path: Union[str, Path, pd.DataFrame], market_tz: Optional[str] = None, **io_kwargs) -> pd.DataFrame:
        """
        Load and standardize an equity dataset.

        Steps
        -----
        1) Read CSV.
        2) Normalize header aliases (e.g., date→timestamp).
        3) Build a datetime index ('timestamp').
        4) Print a validation summary.

        Parameters
        ----------
        df_or_path : str | Path | DataFrame
            Source of the data.
            
        Returns
        -------
        DataFrame
            Clean, UTC‑indexed equity DataFrame.
        """
        if isinstance(df_or_path, (str, Path)):
            df = read_tabular(df_or_path, **io_kwargs)
        else:
            df = df_or_path.copy()
        df = normalize_column_aliases(df)
        df = ensure_datetime_index(df)

        validate(df, ["open","high","low","close","volume"], name="Equity")
        return df


    def interpolate_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill small gaps using the configured method and max_gap.
        """
        return interpolate_missing(df, method=self.cfg.interpolate_method, max_gap=self.cfg.max_gap)

    def detect_outliers(self, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """
        Append outlier diagnostics (z‑score & IsolationForest).
        """
        x = add_rolling_zscore(df.copy(), price_col, self.cfg.zscore_window)
        x["outlier_z"] = x[f"{price_col}_z"].abs() > self.cfg.zscore_threshold
        if price_col in x.columns:
            x["ret"] = x[price_col].pct_change()
        cols = [c for c in ["ret", price_col] if c in x.columns]
        if cols:
            clean = x[cols].dropna()
            if not clean.empty:
                model = IsolationForest(contamination=self.cfg.iforest_contamination,
                                        random_state=self.cfg.iforest_random_state)
                model.fit(clean)
                preds = pd.Series(model.predict(clean), index=clean.index).eq(-1)
                x["outlier_iforest"] = False
                x.loc[preds.index, "outlier_iforest"] = preds
            else:
                x["outlier_iforest"] = False
        return x


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def save_csv(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """
    Save the DataFrame to CSV. The timestamp index is preserved and labeled.

    Notes
    -----
    - We do NOT reset the index—downstream tools often expect a time index.
    - If your consumer needs a column instead, you can reset_index() after load.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index_label="timestamp")
    return out_path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _default_output_path(input_path: Union[str, Path]) -> Path:
    """
    Produce a sensible default output file name: '<stem>_processed.csv'.
    """
    p = Path(input_path)
    return p.with_name(f"{p.stem}_processed.csv")

def main():
    """
    Command‑line entry point.

    Steps performed:
      1) Load equities from --input (CSV),
      2) Resample (--freq), interpolate (--interpolate), and compute outlier flags.
      3) Save to --output (CSV). If omitted, uses '<input>_processed.csv'.
    """
       
    parser = argparse.ArgumentParser(description="Process equity file and save CSV output.")
    parser.add_argument("--input", "-i", default="input_data\COP (Conoco Phillips).csv", help="Path to input equities file (CSV).")
    parser.add_argument("--output", "-o", help="Path to output CSV (default: <input>_processed.csv).")
    parser.add_argument("--price-col", default="close", help="Column used for outlier detection.")
    parser.add_argument("--freq", default="1D", help="Resample frequency, e.g. '1D'.")
    parser.add_argument("--interpolate", default="time", choices=["time","linear","ffill"], help="Interpolation method.")
    args = parser.parse_args()
        

    cfg = EquityConfig(output_freq=args.freq, interpolate_method=args.interpolate)
    pipe = EquityPipeline(cfg)


    # 1) Load & standardize
    eq = pipe.load(args.input)


    # 2) Standardization pipeline (resample → interpolate → outliers)
    eq = pipe.interpolate_missing(eq)
    eq = pipe.detect_outliers(eq, price_col=args.price_col)

    # 4) Persist
    out_path = Path(args.output) if args.output else _default_output_path(args.input)
    save_csv(eq, out_path)
    print(f"[DONE] Saved {len(eq):,} rows to: {out_path}")

if __name__ == "__main__":
    main()

#python equity_processing_pipeline.py --input "input_data/COP (Conoco Phillips).csv" --output processing_output/cop_processed.csv