"""
equity_feature_engineering.py
-----------------------------

Cross-Asset Feature Generation + Dimensionality Reduction & Risk Modeling

Implements
1) Cross–Asset Feature Generation
   (a) Rolling correlations between asset classes
   (b) Volatility regime indicators via Hidden Markov Models (HMM)
   (c) Macro-economic factors (yield-curve slope, simple Fed-sentiment proxy)
   (d) Flow-based features (ETF flows, basic options flow aggregates)

2) Dimensionality Reduction & Risk Modeling
   (a) PCA (risk reduction, low-rank covariances)
   (b) Dynamic correlation modeling (DCC-GARCH, light implementation)
   (c) Regime switching (bull/bear) on returns
   (d) Copula models (Gaussian copula + tail dependence proxies)
   (e) Network analysis (correlation network, MST, centrality)

Design
- Feed-agnostic: you pass DataFrames for prices, volumes, yields, options, etc.
- Minimal dependencies by default; optional blocks activate if packages exist.
- Returns tidy feature DataFrames keyed by a common datetime index.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import os
import argparse

# -------------------- Optional dependencies (graceful degrade) --------------------
# If these are not installed, the relevant blocks will return NaNs or use fallbacks.

try:
    from hmmlearn.hmm import GaussianHMM
    _HAS_HMM = True
except Exception:
    _HAS_HMM = False

try:
    # We use arch for univariate GARCH. The DCC recursion is implemented here directly.
    from arch.univariate import ConstantMean, GARCH, Normal
    _HAS_ARCH = True
except Exception:
    _HAS_ARCH = False

try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    _HAS_NX = False

try:
    from scipy.stats import norm, rankdata
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ============================= Utilities & Helpers ==============================

def _ensure_datetime_index(df: pd.DataFrame, col: Optional[str] = None) -> pd.DataFrame:
    """
    Ensure a DataFrame is indexed by a sorted DatetimeIndex.

    Parameters
    ----------
    df : pd.DataFrame
        The input data.
    col : Optional[str]
        If provided and present in df, this column will be parsed as datetime
        and set as the index.

    Returns
    -------
    pd.DataFrame
        Same data indexed by a sorted DatetimeIndex.

    Raises
    ------
    ValueError
        If a DatetimeIndex cannot be established.
    """
    # If a date column is provided, parse it and set as index
    if col and col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col])
        df = df.set_index(col).sort_index()

    # If the index isn't datetime, try to parse it
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
        except Exception as e:
            raise ValueError("Provide a DatetimeIndex or a parsable date column.") from e
    return df


def logret(prices: pd.Series) -> pd.Series:
    """
    Compute log returns from a price series.

    Notes
    -----
    - Log returns are additive across time and more numerically stable for
      compounding operations.
    """
    p = prices.astype(float).replace(0, np.nan).dropna()
    r = np.log(p).diff()
    return r


def rolling_corr_matrix(returns: pd.DataFrame, window: int = 60, min_periods: int = 30) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Compute a rolling correlation matrix over the asset universe.

    Parameters
    ----------
    returns : pd.DataFrame
        Wide DataFrame of returns (columns = assets).
    window : int
        Lookback window (rows) for each correlation matrix.
    min_periods : int
        Minimum rows required to compute a correlation matrix.

    Returns
    -------
    Dict[timestamp, DataFrame]
        Mapping from window-end timestamp to correlation matrix at that time.
    """
    out = {}
    r = returns.dropna(how="all")
    # Iterate over feasible window end points
    for t in r.index[window-1:]:
        wnd = r.loc[:t].tail(window)
        if len(wnd) >= min_periods:
            out[t] = wnd.corr()
    return out


def _avg_pairwise_corr(df: pd.DataFrame, window: int, min_periods: int) -> pd.Series:
    """
    Average pairwise correlation across columns for each rolling window.

    Why this method?
    ----------------
    - `DataFrame.rolling(...).apply(...)` sends a 1D array to the function (per column),
      which makes multi-column correlation awkward and error-prone.
    - Here we explicitly slice the rolling window across the FULL DataFrame at each step,
      compute its correlation matrix, and average the OFF-DIAGONAL entries only.

    Parameters
    ----------
    df : pd.DataFrame
        Wide DataFrame (columns = assets) of returns.
    window : int
        Lookback window (rows).
    min_periods : int
        Minimum observations required for a valid window.

    Returns
    -------
    pd.Series
        Series of the same index as df with the average off-diagonal correlation.
    """
    idx = df.index
    out = np.full(len(df), np.nan, dtype=float)

    for i in range(len(df)):
        # Determine the window start (inclusive) and take the slice
        start = max(0, i - window + 1)
        wnd = df.iloc[start:i+1]

        if len(wnd) < min_periods:
            continue

        # Compute the correlation matrix for the current window
        C = wnd.corr().to_numpy()
        n = C.shape[0]
        if n <= 1:
            continue

        # Mask the diagonal; we only want average pairwise (off-diagonal) correlation
        mask = ~np.eye(n, dtype=bool)
        vals = C[mask]

        # Average the off-diagonal elements, ignoring NaNs
        if vals.size > 0:
            out[i] = np.nanmean(vals)

    return pd.Series(out, index=idx, name=f"avg_pair_corr_{window}d")


# ======================= Cross–Asset Feature Generation =========================

@dataclass
class CrossAssetFeatures:
    """
    Build cross-asset features from price & flow inputs.

    Inputs
    ------
    prices : pd.DataFrame
        Wide price panel (columns = tickers/assets). Must be aligned (same index).
    volumes : Optional[pd.DataFrame]
        Matching volume panel. (Not required for current features, reserved for extensions.)
    etf_flows : Optional[pd.DataFrame]
        Estimated ETF share creations/redemptions or net flow metrics (columns = tickers/ETFs).
    options : Optional[pd.DataFrame]
        Options tape (flexible schema). If present, we expect:
        ['date','underlying','call_put','volume','open_interest','delta','notional'] (subset ok).
    """
    prices: pd.DataFrame
    volumes: Optional[pd.DataFrame] = None
    etf_flows: Optional[pd.DataFrame] = None
    options: Optional[pd.DataFrame] = None

    def _prep(self) -> pd.DataFrame:
        """
        Ensure prices have a proper datetime index and no holes (FFill/BFill).
        """
        p = _ensure_datetime_index(self.prices)
        p = p.sort_index().ffill().bfill()
        return p

    def correlation_features(self, window: int = 60, min_periods: int = 30) -> pd.DataFrame:
        """
        Produce correlation-derived features:
        - Rolling correlation of each asset vs a benchmark (SPY if present else first column)
        - Average pairwise correlation across the universe (off-diagonal) for each date
        """
        p = self._prep()
        # Use simple returns here; log returns are also fine for correlations
        r = p.pct_change().replace([np.inf, -np.inf], np.nan)

        cols = list(r.columns)
        feats = []

        # Select a benchmark: SPY if available, otherwise the first asset
        bench = "SPY" if "SPY" in cols else cols[0]
        rb = r[bench]

        # Rolling correlation to benchmark for each asset
        # Note: apply works column-wise; corr(rb) aligns on index automatically.
        corr_to_bench = r.apply(lambda x: x.rolling(window, min_periods=min_periods).corr(rb))
        corr_to_bench.columns = [f"corr_{c}_to_{bench}_{window}d" for c in corr_to_bench.columns]
        feats.append(corr_to_bench)

        # Average pairwise correlation across assets (robust implementation)
        avg_pair_corr = _avg_pairwise_corr(r, window=window, min_periods=min_periods).to_frame()
        feats.append(avg_pair_corr)

        # Merge feature blocks on the time index
        return pd.concat(feats, axis=1)

    def volatility_regime_hmm(self, n_states: int = 2, feature: str = "abs_ret") -> pd.DataFrame:
        """
        Estimate volatility regimes using a Gaussian HMM on an equal-weight basket's
        absolute or squared returns.

        Returns
        -------
        DataFrame
            Columns: hmm_state_prob_0..(n_states-1), hmm_state (most likely state)
        """
        p = self._prep()
        # Equal-weight basket return (across all assets)
        r = p.pct_change().mean(axis=1).dropna()

        # Feature choice: absolute return (default) or squared return
        x = r.abs() if feature == "abs_ret" else r.pow(2)
        X = x.dropna().to_frame(name="feat").values
        idx = x.dropna().index

        if not _HAS_HMM:
            # If hmmlearn is unavailable, return NaNs but keep the shape/signature
            warnings.warn("hmmlearn not installed; returning NaNs.")
            return pd.DataFrame(index=p.index, data={
                **{f"hmm_state_prob_{i}": np.nan for i in range(n_states)},
                "hmm_state": np.nan
            })

        # Fit a simple Gaussian HMM
        hmm = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=42)
        hmm.fit(X)
        post = hmm.predict_proba(X)            # Posterior probabilities per state
        state = post.argmax(axis=1)            # Most likely state at each time
        out = pd.DataFrame(post, index=idx, columns=[f"hmm_state_prob_{i}" for i in range(n_states)])
        out["hmm_state"] = state

        # Reindex back to full price index and forward fill (state continuity)
        return out.reindex(p.index).ffill()

    def macro_factors(self, yields: Optional[pd.DataFrame] = None,
                      fed_sentiment: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Build macro factors if provided:
        - Yield curve slope (10Y - 2Y) from a yields panel
        - Fed sentiment proxy as a precomputed daily score
        """
        feats = []

        # Yield curve slope: handle flexible column naming
        if yields is not None and isinstance(yields, pd.DataFrame):
            y = _ensure_datetime_index(yields).sort_index().ffill()
            cols = {c.lower(): c for c in y.columns}
            c2 = cols.get("2y") or cols.get("us2y") or None
            c10 = cols.get("10y") or cols.get("us10y") or None
            if c2 and c10:
                slope = (y[c10] - y[c2]).rename("yc_slope_10y_2y")
                feats.append(slope)

        # Fed sentiment: any provided daily score series
        if fed_sentiment is not None and isinstance(fed_sentiment, pd.Series):
            s = _ensure_datetime_index(fed_sentiment.to_frame("fed_sent")).iloc[:, 0]
            feats.append(s.rename("fed_sentiment"))

        if feats:
            return pd.concat(feats, axis=1)
        return pd.DataFrame()

    def flow_features(self) -> pd.DataFrame:
        """
        Create flow-based features:
        - ETF flows: 5D rolling mean smoothing (helps reduce noise)
        - Options flow: total notional by call/put and an imbalance metric
        """
        feats = []

        # ETF flows (already aligned by date). Smooth with a short MA.
        if self.etf_flows is not None:
            ef = _ensure_datetime_index(self.etf_flows).sort_index()
            ef_roll = ef.rolling(5, min_periods=2).mean().add_suffix("_flow_5d_ma")
            feats.append(ef_roll)

        # Options flow aggregation
        if self.options is not None and len(self.options) > 0:
            opt = self.options.copy()

            # Normalize date index if present
            if "date" in opt.columns:
                opt["date"] = pd.to_datetime(opt["date"])
                opt = opt.set_index("date").sort_index()

            # Group by call/put and aggregate notional (or volume as fallback)
            if "call_put" in opt.columns:
                if "notional" not in opt.columns:
                    # Fallback hierarchy: notional → volume → zeros
                    base = opt.get("notional") or opt.get("volume") or 0.0
                    opt["notional"] = base

                agg = opt.groupby([opt.index, "call_put"])["notional"].sum().unstack(fill_value=0.0)
                agg.columns = [f"opt_notional_{c.lower()}" for c in agg.columns]
                # Positive = call-dominant flow; negative = put-dominant flow
                agg["opt_call_put_imbalance"] = agg.get("opt_notional_call", 0.0) - agg.get("opt_notional_put", 0.0)
                feats.append(agg)

        if feats:
            return pd.concat(feats, axis=1)
        return pd.DataFrame()


# ================= Dimensionality Reduction & Risk Modeling =====================

@dataclass
class RiskModeling:
    """
    Risk-modeling utilities that operate on a return panel (wide DataFrame).
    """
    returns: pd.DataFrame  # wide, columns = assets

    def pca(self, n_components: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Run PCA on standardized returns to extract principal components.

        Parameters
        ----------
        n_components : Optional[int]
            Number of PCs. If None, pick a conservative sqrt(N) up to 10.

        Returns
        -------
        scores : pd.DataFrame
            Time series of component scores (one column per PC).
        loadings : pd.DataFrame
            Asset loadings (rows = assets, columns = PCs).
        explained_variance_ratio : np.ndarray
            Fraction of variance explained by each PC.
        """
        from sklearn.decomposition import PCA
        r = self.returns.dropna()

        # Standardize columns (z-score) to avoid scale effects
        X = (r - r.mean()) / (r.std(ddof=0) + 1e-9)

        # Heuristic: sqrt(N) capped at 10, but >= 1
        k = n_components or min(10, max(1, int(np.sqrt(X.shape[1]))))

        p = PCA(n_components=k, random_state=42)
        Z = p.fit_transform(X.values)  # (T x k) component scores

        scores = pd.DataFrame(Z, index=X.index, columns=[f"pc{i+1}" for i in range(k)])
        loadings = pd.DataFrame(p.components_.T, index=X.columns, columns=scores.columns)

        return scores, loadings, p.explained_variance_ratio_

    def dcc_garch(self, p: int = 1, q: int = 1) -> Dict[str, pd.DataFrame]:
        """
        Lightweight DCC-GARCH(1,1)-style estimator.

        Steps
        -----
        1) Fit univariate vol models to each series to get conditional sigmas.
           - If `arch` is available: GARCH(1,1). Otherwise: EWMA(0.94) proxy.
        2) Standardize returns by these sigmas → Z (approx i.i.d. residuals).
        3) Run a simplified DCC recursion on Z with fixed (a=0.01, b=0.98).
        4) Convert dynamic correlation Rt back to covariance Ht using sigmas.

        Returns
        -------
        dict
            "Rc_t": dict[Timestamp → correlation DataFrame]
            "Ht_t": dict[Timestamp → covariance DataFrame]
        """
        r = self.returns.dropna()
        cols = list(r.columns)
        T, N = r.shape

        # --------- Step 1: conditional volatilities for each asset ----------
        cond_sig = pd.DataFrame(index=r.index, columns=cols, dtype=float)

        if _HAS_ARCH:
            # Fit GARCH(1,1) to each asset individually
            for c in cols:
                y = r[c].dropna()
                am = ConstantMean(y)
                am.volatility = GARCH(1, 0, 1)
                am.distribution = Normal()
                res = am.fit(disp="off")
                sig = res.conditional_volatility.reindex(r.index).ffill().bfill()
                cond_sig[c] = sig
        else:
            # EWMA fallback if arch isn't available
            warnings.warn("arch not installed; using EWMA(0.94) volatility proxy.")
            lam = 0.94
            for c in cols:
                x = r[c].fillna(0.0)
                s2 = np.empty(len(x))
                s2[0] = x.var() if np.isfinite(x.var()) else 1e-6
                for t in range(1, len(x)):
                    s2[t] = lam * s2[t-1] + (1-lam) * x.iloc[t-1] ** 2
                cond_sig[c] = np.sqrt(s2)

        # --------- Step 2: standardize returns by conditional volatility -----
        Z = r / (cond_sig.replace(0, np.nan) + 1e-12)
        Z = Z.replace([np.inf, -np.inf], np.nan).dropna()

        # --------- Step 3: DCC recursion (fixed parameters) -----------------
        S = Z.cov().to_numpy()  # unconditional covariance of standardized residuals
        a = 0.01                # DCC "shock" parameter
        b = 0.98                # DCC "decay" parameter (a + b < 1 is typical)
        Q_t = np.zeros((len(Z), N, N))
        Qbar = S
        Q_t[0] = Qbar.copy()

        # Standard scalar DCC(1,1) recursion
        for t in range(1, len(Z)):
            zt1 = Z.iloc[t-1].values.reshape(-1, 1)
            Q_t[t] = (1 - a - b) * Qbar + a * (zt1 @ zt1.T) + b * Q_t[t-1]

        # --------- Step 4: Convert Qt to Rt (correlation) and Ht (covariance)
        Rc_list = []
        Hc_list = []
        dates = Z.index

        for t in range(len(dates)):
            Qt = Q_t[t]

            # Normalize to a correlation matrix: Rt = D^{-1} Qt D^{-1}
            Dinv = np.diag(1.0 / (np.sqrt(np.diag(Qt)) + 1e-12))
            Rt = Dinv @ Qt @ Dinv

            # Back to covariance using conditional sigmas at time t
            sig_t = cond_sig.loc[dates[t]].values
            Dsig = np.diag(sig_t)
            Ht = Dsig @ Rt @ Dsig

            Rc_list.append(pd.DataFrame(Rt, index=cols, columns=cols))
            Hc_list.append(pd.DataFrame(Ht, index=cols, columns=cols))

        Rc_t = {dates[t]: Rc_list[t] for t in range(len(dates))}
        Ht_t = {dates[t]: Hc_list[t] for t in range(len(dates))}
        return {"Rc_t": Rc_t, "Ht_t": Ht_t}

    def regime_switching_returns(self, n_states: int = 2) -> pd.DataFrame:
        """
        HMM-based (or fallback) regime switching on equal-weighted returns.

        Returns
        -------
        DataFrame
            Columns: reg_state_prob_*, reg_state
        """
        # Equal-weight return across assets
        ew = self.returns.mean(axis=1).dropna()
        x = np.c_[ew.values]   # shape (T, 1)
        idx = ew.index

        if _HAS_HMM:
            hmm = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=7)
            hmm.fit(x)
            post = hmm.predict_proba(x)
            state = post.argmax(axis=1)
            out = pd.DataFrame(post, index=idx, columns=[f"reg_state_prob_{i}" for i in range(n_states)])
            out["reg_state"] = state
            return out
        else:
            # Fallback: approximate regimes via volatility tertiles over a 21D window
            warnings.warn("hmmlearn not installed; regime switching falls back to volatility tertiles.")
            vol = ew.rolling(21).std()
            q = pd.qcut(vol.rank(method="first"), q=n_states, labels=False).reindex(idx)
            out = pd.DataFrame(index=idx)
            for i in range(n_states):
                out[f"reg_state_prob_{i}"] = (q == i).astype(float)
            out["reg_state"] = q
            return out

    def gaussian_copula_features(self) -> Dict[str, pd.DataFrame]:
        """
        Build Gaussian-copula-based dependence features.

        Steps
        -----
        1) Rank → Uniform(0,1) → Inverse-Normal transform (per column) to obtain
           Gaussianized margins Z.
        2) Compute correlation in Z-space (captures nonlinear dependence robustly).
        3) Simple tail co-movement proxies via 95th/5th percent indicators.

        Returns
        -------
        dict
            "Z": Gaussianized series (same index/columns as returns)
            "Corr": static correlation matrix of Z
            "Tail": concatenated upper/lower tail co-occurrence correlations
        """
        r = self.returns.dropna()

        if not _HAS_SCIPY:
            warnings.warn("scipy not installed; copula features limited.")
            return {"Z": r.copy(), "Corr": r.corr(), "Tail": pd.DataFrame()}

        Z = r.copy()
        for c in r.columns:
            # Rank transform (average method for ties)
            ranks = rankdata(r[c].values, method="average")
            # Map to open interval (0,1) and then Gaussianize with probit (norm.ppf)
            u = (ranks - 0.5) / len(ranks)
            Z[c] = norm.ppf(np.clip(u, 1e-6, 1-1e-6))

        Corr = Z.corr()

        # Simple tail-dependence proxies: co-incidence of extreme events
        upper = (Z > Z.quantile(0.95)).astype(int)
        lower = (Z < Z.quantile(0.05)).astype(int)
        tail_up = upper.corr()
        tail_dn = lower.corr()
        tails = pd.concat({"upper_tail_corr": tail_up, "lower_tail_corr": tail_dn}, axis=1)

        return {"Z": Z, "Corr": Corr, "Tail": tails}

    def network_analysis(self, method: str = "mst") -> Dict[str, object]:
        """
        Build a correlation network from the return panel and derive:
        - MST (minimum spanning tree) based on correlation distance
        - Degree centrality (if networkx available)
        - Distance matrix (always returned)

        Distance
        --------
        d_ij = sqrt(2 * (1 - corr_ij))  in [0, 2], where 0 = perfectly correlated.

        Returns
        -------
        dict
            "mst": networkx.Graph (or None if networkx missing)
            "centrality": dict of node → centrality (or empty)
            "distance": pd.DataFrame of distances (always)
        """
        corr = self.returns.corr()
        dist = np.sqrt(2 * (1 - corr.clip(-1, 1)))

        if _HAS_NX:
            # Construct a complete graph weighted by distance
            G = nx.Graph()
            for i in corr.index:
                G.add_node(i)
            for i in corr.index:
                for j in corr.columns:
                    if i < j:
                        G.add_edge(i, j, weight=float(dist.loc[i, j]))
            # Compute MST
            mst = nx.minimum_spanning_tree(G, weight="weight")
            try:
                cent = nx.degree_centrality(mst)
            except Exception:
                cent = {}
            return {"mst": mst, "centrality": cent, "distance": dist}
        else:
            warnings.warn("networkx not installed; returning distance matrix only.")
            return {"mst": None, "centrality": {}, "distance": dist}


# =============================== High-Level Pipeline ============================

@dataclass
class FeatureEngineeringEngine:
    """
    Orchestrator for building the complete feature/risk package.

    Provide as many optional inputs as you have. Modules auto-skip gracefully
    if the corresponding inputs are missing (e.g., no options → no options flow).
    """
    prices: pd.DataFrame
    volumes: Optional[pd.DataFrame] = None
    etf_flows: Optional[pd.DataFrame] = None
    options: Optional[pd.DataFrame] = None
    yields: Optional[pd.DataFrame] = None
    fed_sentiment: Optional[pd.Series] = None

    def run(self) -> Dict[str, object]:
        """
        Execute the full pipeline:
        - Cross-asset features (correlations, regimes, macro, flows)
        - Risk modeling (PCA, DCC, regimes, copula, network)

        Returns
        -------
        dict
            {
              "features_ts":   pd.DataFrame,    # time series features (merged)
              "pca_loadings":  pd.DataFrame,    # static loadings
              "pca_var_ratio": np.ndarray,      # variance explained
              "dcc":           dict,            # dynamic corr/cov matrices
              "copula":        dict,            # copula-based outputs
              "network":       dict,            # MST/centrality/distance
            }
        """
        # Basic cleansing of prices and construction of returns
        P = _ensure_datetime_index(self.prices).sort_index().ffill().bfill()
        R = P.pct_change().replace([np.inf, -np.inf], np.nan)

        # ----- Cross-asset features -----
        caf = CrossAssetFeatures(P, self.volumes, self.etf_flows, self.options)
        corr_feats = caf.correlation_features()      # corr to benchmark + avg pairwise corr
        hmm_vol = caf.volatility_regime_hmm()        # volatility regimes
        macro = caf.macro_factors(self.yields, self.fed_sentiment)  # yield slope + fed sentiment (if provided)
        flows = caf.flow_features()                  # ETF + options flow features

        # ----- Risk modeling -----
        rm = RiskModeling(R)
        pca_scores, pca_loadings, pca_var = rm.pca() # PCA time-series + loadings
        dcc = rm.dcc_garch()                         # dynamic correlations/covariances
        regimes = rm.regime_switching_returns()      # regimes on equal-weighted returns
        cop = rm.gaussian_copula_features()          # copula outputs
        net = rm.network_analysis()                   # network outputs

        # Merge all time-series features on index (some blocks may be empty)
        ts_feats = [corr_feats, hmm_vol, macro, flows, pca_scores, regimes]
        ts_feats = [f for f in ts_feats if isinstance(f, pd.DataFrame) and len(f) > 0]
        F = pd.concat(ts_feats, axis=1).sort_index()

        return {
            "features_ts": F,
            "pca_loadings": pca_loadings,
            "pca_var_ratio": pca_var,
            "dcc": dcc,
            "copula": cop,
            "network": net,
        }


def main():
    """
    Command-line usage example.

    Expected input CSV columns (flexible):
      - Either a 'Date' column or index already as dates
      - 'Close' (or 'Adj Close' or 'close') price
      - Optional 'Volume'

    The demo synthesizes a 3-asset panel (COP + 2 synthetic companions) so that
    cross-asset features make sense even if you only supply one file.
    """
    p = argparse.ArgumentParser(description="Feature Engineering Engine demo")
    p.add_argument(
        "-i", "--input",
        required=False,
        default="input_data/COP (Conoco Phillips).csv",
        help="Input CSV file path (must contain a price column like Close/Adj Close)."
    )
    args = p.parse_args()

    sample_path = args.input
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)

        # --------- Identify price and date columns (flexible naming) ----------
        price_col = None
        for c in df.columns:
            lc = c.lower()
            if lc in {"close", "adj close", "price", "last"}:
                price_col = c
                break

        date_col = None
        for c in df.columns:
            if c.lower() in {"date", "time", "timestamp"}:
                date_col = c
                break

        if price_col is None:
            raise ValueError("Could not find a price column (Close/Adj Close/Price).")

        # Create a proper datetime index (from column or index)
        if date_col is None:
            df = _ensure_datetime_index(df)
        else:
            df = _ensure_datetime_index(df, date_col)

        # Rename to a canonical ticker label for consistency
        cop = df[[price_col]].rename(columns={price_col: "COP"})

        # ---------- Build a small multi-asset panel for demonstration ----------
        # Synthetic companions derived from COP to simulate cross-asset relations.
        spy = (cop["COP"] * (1.0 + 0.0005*np.arange(len(cop)))).rename("SPY")
        uso = (cop["COP"] * (1.0 + 0.0002*np.sin(np.arange(len(cop))/10.0))).rename("USO")

        # Wide price panel (columns = assets)
        prices = pd.concat([cop["COP"], spy, uso], axis=1).ffill().bfill()

        # Run the engine
        engine = FeatureEngineeringEngine(prices=prices)
        results = engine.run()

        # ------------------- Print compact previews to console ------------------
        print("\n=== Feature Time-Series (head) ===")
        print(results["features_ts"].dropna(how="all").head())

        print("\n=== PCA Loadings (head) ===")
        print(results["pca_loadings"].head())

        print("\nExplained Variance Ratio:", results["pca_var_ratio"])

        # DCC: print the most recent correlation matrix if available
        dcc_R = results["dcc"]["Rc_t"]
        if len(dcc_R) > 0:
            last_t = sorted(dcc_R.keys())[-1]
            print(f"\n=== DCC Correlation @ {last_t.date()} ===")
            print(dcc_R[last_t])

        print("\n=== Copula Corr (head) ===")
        print(results["copula"]["Corr"].head())

        print("\n=== Network distance matrix (head) ===")
        print(results["network"]["distance"].head())
    else:
        # Friendly message if the file isn't found
        print("Sample CSV not found. Provide your price panel as a wide DataFrame and call FeatureEngineeringEngine(prices=...).run()")

if __name__ == "__main__":
    # Example direct run:
    # python equity_feature_engineering.py -i "input_data/COP (Conoco Phillips).csv" 
    main()