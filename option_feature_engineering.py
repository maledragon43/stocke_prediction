#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_engineering_engine.py
-----------------------------
Feature Engineering Engine for options data (CSV).

It computes, per underlying symbol:
(1) Asset-Specific Features
    - Liquid expiration buckets: nearest to ~1w / ~1m / ~3m relative to "today"
    - IV term structure from ATM proxies: level, slope, curvature (quadratic fit)
    - Greeks: ATM Greeks averaged across chosen buckets (Δ, Γ, ν/vega, θ, ρ)

(2) Cross-Asset Feature Generation (best-effort with options-only input)
    - Flow aggregates: total volume, total open interest (if present)
    - (Optional) Time-series features such as HMM would require historical index

(3) Dimensionality Reduction & Risk Modeling
    - PCA across [IV level/slope/curvature, flows, ATM Greeks]
    - Correlation network via a minimum spanning tree (MST) built from
      engineered features (requires networkx; skipped if not installed)

Design
- Resilient to missing columns: optional blocks auto-skip if inputs absent.
- ATM selection prefers |abs(delta)-0.5| minimal row; falls back to strike median.
- Expirations are bucketed by nearest target in days: [7, 30, 90].

I/O
- Reads a CSV with flexible headers (case-insensitive aliases).
- Writes 3 outputs into an output directory (default: /mnt/data):
  1) engineered_features.csv           : per-symbol engineered features (+ PCs)
  2) feature_pca_components.csv        : PCA loadings (if PCA ran)
  3) corr_network_edges.csv            : MST edges (if networkx available)

Usage
------
python feature_engineering_engine.py \
    -i input_data/options.csv \
    -o /mnt/data

Notes
- If you pass a different filename, it's fine; column aliasing is handled.
- If expirations are in the past, they are ignored (days_to_exp >= 0).
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import argparse
import os

import numpy as np
import pandas as pd

# Optional dependencies are guarded; the engine continues if they are absent.
try:
    import networkx as nx  # used for correlation MST
    _HAVE_NX = True
except Exception:
    _HAVE_NX = False

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Target expiries for bucketing in days (~1 week, ~1 month, ~3 months)
TARGET_BUCKET_DAYS = np.array([7, 30, 90])  # used by _bucket_label()


# ----------------------- Utilities -----------------------

def _coalesce_columns(df: pd.DataFrame, name: str, aliases: List[str]) -> Optional[str]:
    """
    Return the first matching column in df that equals `name` or one of `aliases`
    (case-insensitive). Returns None if none found.
    """
    cols = {c.lower(): c for c in df.columns}
    if name.lower() in cols:
        return cols[name.lower()]
    for a in aliases:
        if a.lower() in cols:
            return cols[a.lower()]
    return None


def _parse_dates_safe(s: pd.Series) -> pd.Series:
    """Parse a Series as datetimes, coercing errors to NaT (no exception)."""
    return pd.to_datetime(s, errors="coerce", utc=False)


def _now_date(df: pd.DataFrame, ts_col: Optional[str]) -> pd.Timestamp:
    """
    Use the max timestamp from `ts_col` (if present) as the reference "today".
    Otherwise, use the current system date. Normalized to midnight.
    """
    if ts_col is not None:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
        if ts.notna().any():
            return ts.max().normalize()
    return pd.Timestamp.today().normalize()


def _ttm_days(expirations: pd.Series, today: pd.Timestamp) -> pd.Series:
    """Compute time-to-maturity in days as (expiration - today)."""
    return (pd.to_datetime(expirations).dt.normalize() - today).dt.days


def _bucket_label(d: float) -> str:
    """
    Map a maturity (in days) to the nearest target bucket among [7, 30, 90].
    Returns a string label in {"1w","1m","3m"}.
    """
    idx = (np.abs(TARGET_BUCKET_DAYS - d)).argmin()
    return {0: "1w", 1: "1m", 2: "3m"}[idx]


def _select_atm_rows(g: pd.DataFrame, delta_col: Optional[str]) -> pd.DataFrame:
    """
    Select a single "ATM" row from a group (symbol, expiration).
    Priority: minimize |abs(delta)-0.5| when delta exists, else choose strike
    closest to the median strike across contracts for that expiration.
    """
    # 1) Prefer delta-based ATM
    if delta_col and delta_col in g.columns and g[delta_col].notna().any():
        sel = g.copy()
        sel["atm_score"] = np.abs(np.abs(pd.to_numeric(sel[delta_col], errors="coerce")) - 0.5)
        sel = sel.sort_values(["atm_score"]).head(1)
        return sel.drop(columns=["atm_score"])

    # 2) Fallback: pick the strike closest to median strike
    strike_col = _coalesce_columns(g, "strike", [])
    if strike_col:
        sel = g.copy()
        medk = pd.to_numeric(sel[strike_col], errors="coerce").median()
        sel["atm_score"] = np.abs(pd.to_numeric(sel[strike_col], errors="coerce") - medk)
        sel = sel.sort_values(["atm_score"]).head(1)
        return sel.drop(columns=["atm_score"])

    # 3) Last resort: take the first row
    return g.head(1)


def _poly_features(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit y = a + b x + c x^2 (quadratic) and return (level=a, slope=b, curvature=c).
    Returns (nan, nan, nan) if inputs are insufficient.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2 or len(ys) < 2 or np.isnan(xs).any() or np.isnan(ys).any():
        return (np.nan, np.nan, np.nan)
    try:
        coefs = np.polyfit(xs, ys, 2)  # [c, b, a]
        c, b, a = coefs
        return (a, b, c)
    except Exception:
        return (np.nan, np.nan, np.nan)


# ----------------------- Core Engine -----------------------

@dataclass
class EngineConfig:
    input_path: str
    # Output directory; files are created inside it.
    out_dir: str = "/mnt/data"
    # Minimum symbols required for cross-asset style modules to be meaningful
    min_symbols_for_xasset: int = 3


class FeatureEngineeringEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        # These will be filled during load() for consistent access elsewhere
        self.delta_col: Optional[str] = None
        self.iv_col: Optional[str] = None
        self.gamma_col: Optional[str] = None
        self.vega_col: Optional[str] = None
        self.theta_col: Optional[str] = None
        self.rho_col: Optional[str] = None
        self.vol_col: Optional[str] = None
        self.oi_col: Optional[str] = None
        self.ts_col: Optional[str] = None

    # ----------------------- Data Loading -----------------------

    def load(self) -> pd.DataFrame:
        """
        Load CSV and normalize columns:
        - required: symbol, expiration
        - optional: delta, implied_volatility, greeks, volume, open_interest, timestamp
        """
        df = pd.read_csv(self.cfg.input_path)

        # Normalize essential columns (case-insensitive, with aliases)
        sym_col = _coalesce_columns(df, "symbol", ["underlying", "ticker", "root"])
        exp_col = _coalesce_columns(df, "expiration", ["expiry", "maturity", "exp_date"])
        if sym_col is None or exp_col is None:
            raise ValueError("Required columns not found: need 'symbol' and 'expiration'.")

        df = df.rename(columns={sym_col: "symbol", exp_col: "expiration"})

        # Resolve optional columns and store their canonical names for later
        self.delta_col = _coalesce_columns(df, "delta", ["Delta"])
        self.iv_col = _coalesce_columns(df, "implied_volatility", ["iv", "IV", "impl_vol", "impliedVol"])
        self.gamma_col = _coalesce_columns(df, "gamma", ["Gamma"])
        self.vega_col = _coalesce_columns(df, "vega", ["Vega", "nu"])
        self.theta_col = _coalesce_columns(df, "theta", ["Theta"])
        self.rho_col = _coalesce_columns(df, "rho", ["Rho"])
        self.vol_col = _coalesce_columns(df, "volume", ["Volume"])
        self.oi_col = _coalesce_columns(df, "open_interest", ["openInterest", "OI"])
        self.ts_col = _coalesce_columns(df, "timestamp", ["quote_time", "date", "datetime", "time"])

        # Parse dates; any failures become NaT (and are safely ignored later)
        df["expiration"] = _parse_dates_safe(df["expiration"])
        if self.ts_col is not None:
            df[self.ts_col] = _parse_dates_safe(df[self.ts_col])

        return df

    # ----------------------- Feature Blocks -----------------------

    def build_term_buckets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        - Compute days_to_exp from 'today' (based on latest timestamp if present).
        - Drop expired contracts.
        - Assign each row to a nearest bucket label in {"1w","1m","3m"}.
        """
        today = _now_date(df, self.ts_col)
        df = df.copy()
        df["days_to_exp"] = _ttm_days(df["expiration"], today)
        df = df[df["days_to_exp"] >= 0]  # ignore expired contracts
        df["bucket"] = df["days_to_exp"].apply(_bucket_label)
        return df

    def compute_atm_iv_by_bucket(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For every (symbol, expiration):
        - Select a single "ATM" row (delta≈0.5 if available, else strike-median proxy).
        - Extract implied vol and greeks for that row.
        - Attach the bucket label computed previously.
        Returns a thin DataFrame with columns:
        ['symbol','expiration','bucket','iv_atm','delta_atm','gamma_atm','vega_atm','theta_atm','rho_atm']
        """
        rows = []
        for (sym, exp), g in df.groupby(["symbol", "expiration"]):
            sel = _select_atm_rows(g, self.delta_col)
            sel = sel.assign(symbol=sym, expiration=exp)
            rows.append(sel.iloc[0])
        atm = pd.DataFrame(rows)

        # Implied vol at ATM (or NaN if unavailable)
        if self.iv_col is None or self.iv_col not in atm.columns:
            warnings.warn("Implied volatility column not found; term-structure features will be NaN.")
            atm["iv_atm"] = np.nan
        else:
            atm["iv_atm"] = pd.to_numeric(atm[self.iv_col], errors="coerce")

        # Greeks at ATM (convert to numeric; if absent -> NaN)
        for col, name in [
            (self.delta_col, "delta_atm"),
            (self.gamma_col, "gamma_atm"),
            (self.vega_col,  "vega_atm"),
            (self.theta_col, "theta_atm"),
            (self.rho_col,   "rho_atm"),
        ]:
            if col and col in atm.columns:
                atm[name] = pd.to_numeric(atm[col], errors="coerce")
            else:
                atm[name] = np.nan

        # Attach bucket labels if not already present
        if "bucket" not in atm.columns and "bucket" in df.columns:
            atm = atm.merge(
                df[["symbol", "expiration", "bucket"]].drop_duplicates(),
                on=["symbol", "expiration"],
                how="left",
            )
        elif "bucket" not in atm.columns:
            atm["bucket"] = np.nan

        return atm[
            ["symbol", "expiration", "bucket",
             "iv_atm", "delta_atm", "gamma_atm", "vega_atm", "theta_atm", "rho_atm"]
        ]

    def aggregate_by_symbol(self, atm: pd.DataFrame, df_all: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate per-symbol features:
        - Pivot ATM implied vols by bucket: iv_atm_1w, iv_atm_1m, iv_atm_3m
        - Average ATM greeks across buckets
        - Sum flows (volume, open_interest) across all rows (if present)
        - Fit quadratic over x = sqrt(T) for (level, slope, curvature)
        """
        # 1) Pivot IVs by bucket (mean across expirations within bucket)
        piv = atm.pivot_table(index="symbol", columns="bucket", values="iv_atm", aggfunc="mean")
        piv.columns = [f"iv_atm_{c}" for c in piv.columns]
        piv = piv.reset_index()

        # 2) Average ATM Greeks across buckets
        greek_cols = ["delta_atm", "gamma_atm", "vega_atm", "theta_atm", "rho_atm"]
        g_agg = (
            atm.groupby("symbol")[greek_cols]
            .mean()
            .reset_index()
            .rename(columns={c: f"{c}_mean" for c in greek_cols})
        )

        # 3) Flows (optional)
        flows = []
        if self.vol_col and self.vol_col in df_all.columns:
            flows.append(df_all.groupby("symbol")[self.vol_col].sum().rename("volume_sum"))
        if self.oi_col and self.oi_col in df_all.columns:
            flows.append(df_all.groupby("symbol")[self.oi_col].sum().rename("open_interest_sum"))
        if flows:
            flow_df = pd.concat(flows, axis=1).reset_index()
        else:
            # Ensure we still have a frame to merge on `symbol`
            flow_df = pd.DataFrame({"symbol": atm["symbol"].unique()})

        # 4) Term-structure shape via quadratic over x = sqrt(T_years)
        bucket_days = {"1w": 7, "1m": 30, "3m": 90}
        coefs = []
        piv_idxed = piv.set_index("symbol")
        for sym, row in piv_idxed.iterrows():
            xs, ys = [], []
            for b, d in bucket_days.items():
                col = f"iv_atm_{b}"
                if col in piv_idxed.columns:
                    val = row.get(col, np.nan)
                    if pd.notna(val):
                        xs.append(np.sqrt(d / 365.25))
                        ys.append(val)
            level, slope, curvature = _poly_features(np.array(xs), np.array(ys))
            coefs.append({"symbol": sym, "iv_level": level, "iv_slope": slope, "iv_curvature": curvature})
        coefs_df = pd.DataFrame(coefs)

        # Merge all parts
        out = (
            piv
            .merge(g_agg, on="symbol", how="outer")
            .merge(flow_df, on="symbol", how="outer")
            .merge(coefs_df, on="symbol", how="outer")
        )
        return out

    def pca_over_symbols(self, feat: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Run PCA across numeric columns and append PCs to the per-symbol table.
        Returns (features_with_PCs, components) where `components` are the loadings.
        Skips if not enough rows/columns.
        """
        num = feat.select_dtypes(include=[np.number]).copy()
        if num.shape[0] < 2 or num.shape[1] < 2:
            return feat, None

        num = num.fillna(num.mean())  # simple mean imputation for NaNs
        scaler = StandardScaler()
        X = scaler.fit_transform(num.values)

        pca = PCA(n_components=min(5, X.shape[1]))
        Z = pca.fit_transform(X)

        # Attach PCs back onto the feature table
        feat = feat.copy()
        for i in range(Z.shape[1]):
            feat[f"PC{i+1}"] = Z[:, i]

        # Component loadings (rows=original numeric features, cols=PCs)
        comp = pd.DataFrame(
            pca.components_.T,
            index=num.columns,
            columns=[f"PC{i+1}" for i in range(Z.shape[1])],
        )
        return feat, comp

    def correlation_network(self, feat: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Build a correlation-based MST over symbols using engineered numeric features.
        Distance = 1 - |corr| between per-symbol feature vectors.
        Requires networkx; returns None if not available or if data is too small.
        """
        if not _HAVE_NX:
            return None
        if "symbol" not in feat.columns:
            return None

        num = feat.set_index("symbol").select_dtypes(include=[np.number]).copy()
        syms = list(num.index)
        if num.shape[1] < 2 or len(syms) < 3:
            return None

        # Build complete graph with distances = 1 - |corr|
        G = nx.Graph()
        for s in syms:
            G.add_node(s)

        arr = num.values
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                v1 = arr[i]
                v2 = arr[j]
                c = np.corrcoef(v1, v2)[0, 1]
                d = 1.0 - abs(c) if np.isfinite(c) else 1.0
                G.add_edge(syms[i], syms[j], weight=d, corr=c)

        T = nx.minimum_spanning_tree(G, weight="weight")
        rows = [{"u": u, "v": v, "distance": data["weight"], "corr": data.get("corr", np.nan)}
                for u, v, data in T.edges(data=True)]
        return pd.DataFrame(rows) if rows else None

    # ----------------------- Orchestration -----------------------

    def run(self) -> Dict[str, Optional[pd.DataFrame]]:
        """Run the full pipeline and return a dict of DataFrames."""
        df = self.load()
        df = self.build_term_buckets(df)
        atm = self.compute_atm_iv_by_bucket(df)
        per_symbol = self.aggregate_by_symbol(atm, df)
        per_symbol_pca, comp = self.pca_over_symbols(per_symbol)
        mst = self.correlation_network(per_symbol_pca)


        # Write outputs
        features_path = os.path.join(self.cfg.out_dir, "option_engineered_features.csv")
        pca_path = os.path.join(self.cfg.out_dir, "feature_pca_components.csv")
        mst_path = os.path.join(self.cfg.out_dir, "corr_network_edges.csv")

        per_symbol_pca.to_csv(features_path, index=False)
        print(f"Wrote features to: {features_path}")
        
        if comp is not None:
            comp.to_csv(pca_path)
        if mst is not None:
            mst.to_csv(mst_path, index=False)

        return {
            "features": per_symbol_pca,
            "pca_components": comp,
            "mst_edges": mst,
        }


# ----------------------- CLI -----------------------

def main():
    parser = argparse.ArgumentParser(description="Options Feature Engineering Engine")
    parser.add_argument(
        "-i", "--input",
        required=False,
        default="input_data/options.csv",
        help="Path to input CSV (options table). Column names are case-insensitive; aliases are supported."
    )

    args = parser.parse_args()

    cfg = EngineConfig(input_path=args.input, out_dir="feature_output")
    eng = FeatureEngineeringEngine(cfg)

    # Run the engine
    out = eng.run()


if __name__ == "__main__":
    
    main()
    #python option_feature_engineering.py -i "input_data/options.csv"