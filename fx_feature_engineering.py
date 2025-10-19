"""
fx_feature_engining.py
----------------------

MAIN BUILDING BLOCKS
(1) Asset-Specific FX Features
    • Pair alignment to rebalance cadence (W/M)                   [NOTE] This file assumes
      the index is already aligned. The advanced version can map raw dates to the last
      weekly/monthly observation before rebalancing. 
    • Carry (rate differential)                                   [requires policy_rates]
    • Momentum (multi-lookback sums of log-returns)
    • Realized volatility (annualized, rolling)
    • Risk-on/off flags via a risk proxy (e.g., SPX)              [optional input]

(2) Cross–Asset Feature Generation
    • Rolling correlations vs. provided cross-asset series
    • Volatility regime indicators via HMM on per-pair returns    [if `hmmlearn` present]
    • Macro factors: yield-curve slope (10y-2y), Fed sentiment z  [if inputs provided]
    • Flow-based features: ETF flows & options-flow aggregates    [if provided]

(3) Dimensionality Reduction & Risk Modeling
    • PCA on standardized features (sklearn)
    • DCC-like dynamic correlation (EWMA proxy) between FX pairs
    • Regime switching probability (Markov switching if available, else heuristic)
    • Copula tail metrics from Gaussian-score transforms (SciPy)
    • Network MST from correlation distances (SciPy)

USAGE
- See the `main()` function at the bottom for a minimal CLI example:
    python fx_feature_engining.py -i input_data/USD_EUR_daily_full.csv
- The output is written as a single CSV joining the key feature blocks.

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import argparse

# --- Optional imports (graceful degradation) -----------------------------------
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    PCA = None
    StandardScaler = None

try:
    from hmmlearn.hmm import GaussianHMM  # Hidden Markov Model for volatility regimes
except Exception:
    GaussianHMM = None

try:
    # Markov switching regression (used for regime probabilities)
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
except Exception:
    MarkovRegression = None

try:
    # SciPy bits for copula and network analysis
    from scipy.stats import norm, rankdata
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import squareform
    from scipy.spatial.distance import pdist
except Exception:
    norm = None
    rankdata = None
    minimum_spanning_tree = None
    squareform = None
    pdist = None

# ----------------------- Utilities ---------------------------------------------
def _as_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a DataFrame is indexed by a proper DatetimeIndex.

    Behavior:
    1) If the current index is not a DatetimeIndex, try common date-like columns:
       'date', 'Date', 'DATE', 'timestamp', 'time'. If found, set as index.
    2) As a last resort, try converting the current index to datetime.
    3) Drop rows with NaT indices and sort chronologically.

    This lets callers pass CSVs with a date column or with the date already
    as the index—both will be handled.
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        for c in ['date', 'Date', 'DATE', 'timestamp', 'time']:
            if c in out.columns:
                out = out.set_index(pd.to_datetime(out[c], errors='coerce')).drop(columns=[c])
                break
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, errors='coerce')
    out = out[~out.index.isna()]
    out = out.sort_index()
    return out

def _pct_ret(x: pd.Series) -> pd.Series:
    """
    Compute log returns: r_t = ln(P_t) - ln(P_{t-1}).
    Using log returns simplifies additive aggregations (e.g., momentum sums).
    """
    return np.log(x).diff()

def _ann_factor(freq: str) -> float:
    """
    Annualization factor for volatility based on sampling frequency.
    - daily  -> sqrt(252)
    - weekly -> sqrt(52)
    - monthly-> sqrt(12)
    Defaults to daily if not recognized.
    """
    freq = (freq or '').lower()
    if freq in ('d','day','daily'):
        return np.sqrt(252.0)
    if freq in ('w','week','weekly'):
        return np.sqrt(52.0)
    if freq in ('m','mo','month','monthly'):
        return np.sqrt(12.0)
    return np.sqrt(252.0)

def _std_rolling(x: pd.Series, win: int) -> pd.Series:
    """Convenience wrapper for rolling standard deviation."""
    return x.rolling(win).std()

def _zscore(x: pd.Series, win: int) -> pd.Series:
    """
    Rolling z-score: (x - mean_win) / std_win.
    Useful for normalizing changes (e.g., policy-rate deltas).
    """
    mu = x.rolling(win).mean()
    sd = x.rolling(win).std()
    return (x - mu) / sd.replace(0, np.nan)

# ----------------------- Core Engine -------------------------------------------
@dataclass
class FXFeatureEngine:
    """
    Main orchestrator class.

    Parameters
    ----------
    rebalance_freq : {"W", "M"}
        Target cadence for features (weekly or monthly). This script assumes
        the input index is already at the desired cadence. If you ingest high
        frequency or daily data but want week/month end alignment, add an
        `_align_to_rebalance(...)` step before feature computation.
    lookbacks : dict
        Window lengths for momentum, volatility, correlations, and flow sums.
        Example: {"mom":[5,21,63], "vol":[21,63], "corr":[21,63], "flows":[5,21]}
    g10_pairs / em_pairs : List[str]
        Lists of tickers to include or track. (Not strictly required for the
        feature formulas; useful for validation & scope checks).
    """
    rebalance_freq: str = "W"
    lookbacks: Dict[str, List[int]] = field(default_factory=lambda: {
        "mom":[5,21,63], "vol":[21,63], "corr":[21,63], "flows":[5,21]
    })
    g10_pairs: List[str] = field(default_factory=lambda: [
        "EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","NZDUSD","USDCAD","EURJPY","EURGBP","EURCHF"
    ])
    em_pairs: List[str] = field(default_factory=list)

    # ---------- Helpers to standardize inputs ----------------------------------
    def _prepare_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize an FX price panel:
        - Datetime index
        - Forward-fill missing values (market holidays, sparse pairs)
        - Drop columns that are all-NaN
        Expect columns named by pairs, e.g., 'EURUSD', 'USDJPY', ...
        """
        px = _as_dt_index(prices)
        px = px.sort_index().ffill()
        px = px.dropna(axis=1, how='all')
        return px

    # ---------- (1) Asset-Specific FX Features --------------------------------
    def _compute_basic_fx(self, prices: pd.DataFrame,
                          policy_rates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute core FX features for each pair:
        - Momentum: sum of log-returns over lookbacks
        - Realized volatility: rolling std * sqrt(252)
        - Carry: policy-rate(base) - policy-rate(quote)   [if policy_rates provided]

        Notes:
        • This implementation assumes prices are already sampled at (or near) the
          desired rebalancing cadence. If you need true W/M period-end alignment,
          resample or map timestamps accordingly before calling this.
        """
        px = self._prepare_prices(prices)

        # If you want explicit rebalancing alignment, do it here
        px_rb = px.copy()
        # e.g., px_rb = align_to_week_or_month(px)  # omitted for simplicity
        px_rb = px_rb[~px_rb.index.isna()].groupby(level=0).last()

        # Log-returns panel
        rets = px_rb.apply(_pct_ret)

        feats = {}

        # Momentum features: cumulative log returns over multiple windows
        for L in self.lookbacks.get("mom", [21,63]):
            feats.update({f"mom_{c}_{L}": rets[c].rolling(L).sum() for c in rets.columns})

        # Realized volatility (annualized) over multiple windows
        for L in self.lookbacks.get("vol", [21,63]):
            v = rets.rolling(L).std() * np.sqrt(252.0)
            feats.update({f"rvol_{c}_{L}": v[c] for c in v.columns})

        # Carry = short_rate(base) - short_rate(quote), using policy rates if provided
        if policy_rates is not None and len(policy_rates.columns)>0:
            pr = _as_dt_index(policy_rates).ffill()
            pr_rb = pr.copy()
            # Same note as prices: if you need explicit period mapping, apply it here
            pr_rb = pr_rb[~pr_rb.index.isna()].groupby(level=0).last()

            def carry_for_pair(pair: str) -> pd.Series:
                """
                For a pair like 'EURUSD' (quoted as USD per EUR):
                carry ≈ policy_rate(EUR) - policy_rate(USD).
                If a currency is missing from `policy_rates`, returns NaNs.
                """
                if len(pair) != 6:
                    return pd.Series(index=px_rb.index, dtype=float)
                base = pair[:3]
                quote = pair[3:]
                if base in pr_rb.columns and quote in pr_rb.columns:
                    return pr_rb[base] - pr_rb[quote]
                return pd.Series(index=px_rb.index, dtype=float)

            for c in px_rb.columns:
                feats[f"carry_{c}"] = carry_for_pair(c)

        # Convert feature dict -> DataFrame, time-aligned to px index
        basic = pd.DataFrame(feats).sort_index()
        return basic

    # ---------- (2) Cross-Asset Features --------------------------------------
    def _compute_cross_asset(self,
                             fx_prices: pd.DataFrame,
                             cross_assets: Optional[Dict[str, pd.Series]] = None,
                             risk_proxy: Optional[pd.Series] = None,
                             etf_flows: Optional[pd.DataFrame] = None,
                             options_flow: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build features that tie FX behavior to other assets & flows.

        Inputs
        ------
        cross_assets : dict of {name: Series}
            e.g., {"US10Y": 10y yield, "US2Y": 2y yield, "FEDFUNDS": policy rate, "SPX": equity index}
        risk_proxy   : Series for risk-on/off flags (e.g., SPX or VIX inverse).
        etf_flows    : DataFrame of ETF flow series (cols = tickers/aggregates)
        options_flow : DataFrame of option flow aggregates (cols = features you created)

        Outputs
        -------
        • Correlations between FX returns and cross-asset returns (rolling windows)
        • Volatility regime flags per pair using HMM (if available)
        • Macro slope (10y-2y) and Fed-sentiment z-score
        • Rolling sums of ETF/option flows
        • Risk-on/off flags from risk proxy
        """
        px = self._prepare_prices(fx_prices)
        px_rb = px.copy()
        px_rb = px_rb[~px_rb.index.isna()].groupby(level=0).last()
        fx_ret = px_rb.apply(_pct_ret)

        feats = {}

        # Rolling correlations between FX and cross-asset returns
        if cross_assets:
            ca_df = []
            for k, s in cross_assets.items():
                s2 = _as_dt_index(pd.DataFrame({k: s}))[k].reindex(px_rb.index).ffill()
                ca_df.append(s2)
            ca_df = pd.concat(ca_df, axis=1)
            ca_ret = ca_df.apply(_pct_ret)

            for L in self.lookbacks.get("corr", [21,63]):
                # Pandas trick: rolling corr on aligned panels gives a wide MultiIndex
                roll = fx_ret.rolling(L).corr(ca_ret)
                # Flatten: pick FX row per CA and create a single Series per FX-CA pair
                for fx in fx_ret.columns:
                    for ca in ca_ret.columns:
                        feats[f"corr_{fx}_{ca}_{L}"] = roll.xs(fx, level=0, axis=1)[ca]

        # Volatility regimes via a simple 2-state Gaussian HMM on per-pair returns
        if GaussianHMM is not None:
            hmm_feats = []
            for c in fx_ret.columns:
                x = fx_ret[c].dropna().values.reshape(-1,1)
                # HMM needs enough samples to be meaningful
                if len(x) > 200:
                    try:
                        model = GaussianHMM(n_components=2, covariance_type='full', random_state=42)
                        model.fit(x)
                        states = pd.Series(model.predict(x), index=fx_ret[c].dropna().index, name=f"hmm_state_{c}")
                        # Choose the state with higher |mean| as the "high-vol" regime
                        means = model.means_.flatten()
                        high_state = int(np.argmax(np.abs(means)))
                        reg = (states==high_state).astype(int).reindex(fx_ret.index)
                        hmm_feats.append(reg.rename(f"vol_regime_{c}"))
                    except Exception:
                        # If HMM fails (e.g., singular cov), skip that pair
                        pass
            if hmm_feats:
                hmm_df = pd.concat(hmm_feats, axis=1)
                feats.update({col: hmm_df[col] for col in hmm_df.columns})

        # Risk-on/off flags from a risk proxy (e.g., mean positive returns over a window)
        if risk_proxy is not None:
            r = _as_dt_index(pd.DataFrame({"risk": risk_proxy}))["risk"].reindex(px_rb.index).ffill()
            r_ret = _pct_ret(r).fillna(0.0)
            for L in [21,63]:
                feats[f"risk_on_{L}"] = (r_ret.rolling(L).mean() > 0).astype(int)

        # Macro factors from provided cross-assets:
        # - yield-curve slope (10y - 2y): requires keys like "US10Y" and "US2Y"
        # - Fed sentiment: z-score of changes in a policy rate series (e.g., "FEDFUNDS")
        if cross_assets:
            ten = None; two = None; ffr = None
            for k in cross_assets.keys():
                key = k.upper()
                if "10" in key and "Y" in key and "US" in key: ten = k
                if ("2" in key or "02" in key) and "Y" in key and "US" in key: two = k
                if "FF" in key or "FEDFUNDS" in key or "POLICY" in key: ffr = k
            if ten and two:
                y10 = _as_dt_index(pd.DataFrame({ten: cross_assets[ten]}))[ten].reindex(px_rb.index).ffill()
                y02 = _as_dt_index(pd.DataFrame({two: cross_assets[two]}))[two].reindex(px_rb.index).ffill()
                slope = (y10 - y02).rename("yc_slope")
                feats["yc_slope"] = slope
            if ffr:
                ff = _as_dt_index(pd.DataFrame({ffr: cross_assets[ffr]}))[ffr].reindex(px_rb.index).ffill()
                ff_chg = ff.diff()
                feats["fed_sentiment_z21"] = _zscore(ff_chg, 21)

        # Flow-based features: rolling sums over chosen windows
        if etf_flows is not None and len(etf_flows.columns)>0:
            flows = _as_dt_index(etf_flows).reindex(px_rb.index).fillna(0.0)
            for L in self.lookbacks.get("flows",[5,21]):
                s = flows.rolling(L).sum()
                for c in s.columns:
                    feats[f"etf_flow_{c}_{L}"] = s[c]

        if options_flow is not None and len(options_flow.columns)>0:
            of = _as_dt_index(options_flow).reindex(px_rb.index).fillna(0.0)
            for L in self.lookbacks.get("flows",[5,21]):
                s = of.rolling(L).sum()
                for c in s.columns:
                    feats[f"opt_flow_{c}_{L}"] = s[c]

        return pd.DataFrame(feats).sort_index()

    # ---------- (3) Dimensionality Reduction & Risk ----------------------------
    def _pca_block(self, X: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        """
        Run PCA on standardized features.
        - Drops all-NaN rows, ffill+bfill missing values, standardizes columns.
        - Caps component count to available columns and to 80% of sample size to
          avoid overfitting in small samples.
        - Returns principal components aligned back to the original index.
        """
        if PCA is None or StandardScaler is None:
            return pd.DataFrame(index=X.index)
        valid = X.replace([np.inf,-np.inf], np.nan).dropna(how='all')
        valid = valid.fillna(method='ffill').fillna(method='bfill')
        scaler = StandardScaler()
        Z = scaler.fit_transform(valid.values)
        pca = PCA(n_components=min(n_components, Z.shape[1], max(1, int(Z.shape[0]*0.8))))
        pcs = pca.fit_transform(Z)
        cols = [f"pca_{i+1}" for i in range(pcs.shape[1])]
        out = pd.DataFrame(pcs, index=valid.index, columns=cols)
        return out.reindex(X.index)

    def _dcc_like(self, returns: pd.DataFrame, alpha: float = 0.05, beta: float = 0.93) -> Dict[Tuple[str,str], pd.Series]:
        """
        Lightweight DCC proxy using EWMA covariance updates.
        - alpha, beta: GARCH-like weights (alpha + beta < 1 is typical).
        - Returns a dict of pairwise correlation time series keyed by (i, j).
        This is sufficient for many applications without the heavy `arch` package.
        """
        cols = list(returns.columns)
        out = {}
        if len(cols) < 2: return out
        rets = returns.fillna(0.0).values
        T, N = rets.shape
        # Initialize with sample covariance matrix
        S = np.cov(rets.T)
        for i in range(N):
            for j in range(i+1, N):
                cov_t = np.zeros(T)
                var_i = np.zeros(T)
                var_j = np.zeros(T)
                cov_t[0] = S[i,j]; var_i[0] = S[i,i]; var_j[0] = S[j,j]
                for t in range(1,T):
                    cov_t[t] = (1-alpha-beta)*S[i,j] + alpha*rets[t-1,i]*rets[t-1,j] + beta*cov_t[t-1]
                    var_i[t] = (1-alpha-beta)*S[i,i] + alpha*rets[t-1,i]**2 + beta*var_i[t-1]
                    var_j[t] = (1-alpha-beta)*S[j,j] + alpha*rets[t-1,j]**2 + beta*var_j[t-1]
                corr_t = cov_t / (np.sqrt(var_i*var_j) + 1e-12)
                out[(cols[i], cols[j])] = pd.Series(corr_t, index=returns.index, name=f"dcc_{cols[i]}_{cols[j]}")
        return out

    def _regime_switching(self, series: pd.Series) -> pd.Series:
        """
        Compute regime probability (e.g., probability of "risk-on" or "high-return" regime).
        - If statsmodels' MarkovRegression is available, we fit a 2-regime model and return
          the smoothed probability of regime 1.
        - Otherwise, we fall back to a logistic transform of a long/short rolling mean z-score,
          which works surprisingly well as a simple proxy.
        """
        s = series.dropna()
        if len(s) < 100:
            return pd.Series(index=series.index, dtype=float)
        if MarkovRegression is not None:
            try:
                mod = MarkovRegression(s, k_regimes=2, trend='c', switching_variance=True)
                res = mod.fit(disp=False)
                probs = res.smoothed_marginal_probabilities[1]  # probability of regime 1
                return probs.reindex(series.index)
            except Exception:
                pass
        # Heuristic fallback: rolling signal transformed via sigmoid
        r = s.rolling(21).mean()
        z = (r - r.rolling(63).mean()) / (r.rolling(63).std() + 1e-9)
        prob = 1/(1+np.exp(-z))
        return prob.reindex(series.index)

    def _copula_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Copula-based tail dependence proxies.
        - Rank-transform each column to (0,1), then map to Gaussian scores via N^{-1}.
        - For each pair (i, j), compute correlation within lower and upper 5% tails.
        Returns a single-row DataFrame snapshot (latest index) of tail metrics.
        """
        if norm is None or rankdata is None or df.shape[1] < 2:
            return pd.DataFrame(index=df.index)
        U = df.rank(axis=0, method='average', pct=True).clip(1e-6, 1-1e-6)
        Z = pd.DataFrame(norm.ppf(U), index=df.index, columns=df.columns)
        cols = df.columns.tolist()
        feats = {}
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                zi, zj = Z[cols[i]], Z[cols[j]]
                q = 0.05
                mask_low = (zi < norm.ppf(q)) & (zj < norm.ppf(q))
                mask_high = (zi > norm.ppf(1-q)) & (zj > norm.ppf(1-q))
                feats[f"tail_low_corr_{cols[i]}_{cols[j]}"] = zi[mask_low].corr(zj[mask_low])
                feats[f"tail_high_corr_{cols[i]}_{cols[j]}"] = zi[mask_high].corr(zj[mask_high])
        # Package as a one-row DataFrame at the most recent timestamp (if any)
        last_idx = [df.index[-1]] if len(df)>0 else []
        return pd.DataFrame({k: pd.Series(v, index=last_idx) for k,v in feats.items()})

    def _network_mst(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Construct a Minimum Spanning Tree (MST) of the FX correlation network.
        - Distance metric: d(i,j) = sqrt(2 * (1 - corr(i,j)))  ∈ [0, 2]
        - Uses SciPy's sparse MST. Returns an edge list (u, v, distance).
        """
        if minimum_spanning_tree is None or returns.shape[1] < 2:
            return pd.DataFrame(index=returns.index)
        C = returns.corr().clip(-0.999, 0.999)
        D = np.sqrt(2*(1 - C.values))
        mst = minimum_spanning_tree(D)
        cols = returns.columns.tolist()
        edges = []
        coo = mst.tocoo()
        for i, j, w in zip(coo.row, coo.col, coo.data):
            edges.append((cols[i], cols[j], float(w)))
        return pd.DataFrame(edges, columns=["node_u","node_v","distance"])

    # ------------------- Public API --------------------------------------------
    def build_features(self,
                       fx_prices: pd.DataFrame,
                       policy_rates: Optional[pd.DataFrame] = None,
                       cross_assets: Optional[Dict[str, pd.Series]] = None,
                       risk_proxy: Optional[pd.Series] = None,
                       etf_flows: Optional[pd.DataFrame] = None,
                       options_flow: Optional[pd.DataFrame] = None,
                       pca_components: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Generate all feature blocks.

        Returns a dict with keys:
          - 'basic_fx' : carry/momentum/realized-volatility
          - 'cross'    : correlations, HMM regimes, macro, flows, risk flags
          - 'pca'      : PCA factor time series
          - 'dcc'      : DCC-like dynamic correlations (wide, concatenated pairs)
          - 'regime'   : single-column DataFrame of regime probabilities (anchor series)
          - 'copula'   : one-row DataFrame with tail dependence proxies
          - 'network'  : MST edge list capturing interconnectedness

        You can join or select blocks as needed for downstream modeling.
        """
        # (1) Asset-specific block
        basic = self._compute_basic_fx(fx_prices, policy_rates=policy_rates)

        # (2) Cross-asset block
        cross = self._compute_cross_asset(
            fx_prices=fx_prices,
            cross_assets=cross_assets,
            risk_proxy=risk_proxy,
            etf_flows=etf_flows,
            options_flow=options_flow
        )

        # (3a) PCA on union of features
        X = basic.join(cross, how='outer').sort_index()
        pca_df = self._pca_block(X, n_components=pca_components)

        # (3b) DCC-like dynamic correlations across FX pairs
        px = self._prepare_prices(fx_prices)
        px_rb = px.copy()
        px_rb = px_rb[~px_rb.index.isna()].groupby(level=0).last()
        fx_ret = px_rb.apply(_pct_ret)
        dcc_map = self._dcc_like(fx_ret.fillna(0.0))
        dcc_df = pd.concat(dcc_map.values(), axis=1) if dcc_map else pd.DataFrame(index=fx_ret.index)

        # (3c) Regime switching on risk proxy (preferred) or mean basket return
        anchor = _as_dt_index(pd.DataFrame({"risk": risk_proxy}))["risk"].reindex(px_rb.index).ffill() \
                 if risk_proxy is not None else fx_ret.mean(axis=1)
        regime_prob = self._regime_switching(anchor)

        # (3d) Copula tails & (3e) Network MST snapshots (optional)
        copula = self._copula_metrics(fx_ret.dropna(how="any"))
        network = self._network_mst(fx_ret.dropna(how="any"))

        return {
            "basic_fx": basic,
            "cross": cross,
            "pca": pca_df,
            "dcc": dcc_df,
            "regime": pd.DataFrame({"regime_prob": regime_prob}),
            "copula": copula,
            "network": network,
        }

# ------------------- CLI entrypoint --------------------------------------------
def main():
    """
    Minimal command-line interface to run the engine on a single input CSV.

    Example:
        python fx_feature_engining.py -i input_data/USD_EUR_daily_full.csv

    Output:
        Writes a single CSV joining main blocks to:
            feature_output/fx_features_all.csv
        Make sure the 'feature_output' directory exists or adjust the path.
    """
    p = argparse.ArgumentParser(description="Feature Engineering Engine demo")
    p.add_argument(
        "-i", "--input",
        required=False,
        default="input_data/USD_EUR_daily_full.csv",
        help="Input CSV file path"
    )
    args = p.parse_args()

    input_file = args.input

    # 1) Load FX prices panel (DatetimeIndex + pair columns). 
    #    date column (e.g., 'date'), 
    fx_prices = pd.read_csv(input_file, parse_dates=[0], index_col=0)

    # Optional data sources (set to None here; wire them in production as needed)
    etf_flows   = None  # DataFrame indexed by date with ETF flow columns
    options_flow = None # DataFrame indexed by date with options flow aggregations

    # 2) Configure the engine: change rebalance frequency or lookbacks as needed
    engine = FXFeatureEngine(
        rebalance_freq="W",  # or "M" for monthly cadence
        lookbacks={"mom":[5,21,63], "vol":[21,63], "corr":[21,63], "flows":[5,21]},
    )

    # 3) Build feature blocks (policy_rates/cross_assets/risk_proxy can be added)
    blocks = engine.build_features(
        fx_prices=fx_prices,
        etf_flows=etf_flows,
        options_flow=options_flow,
        pca_components=5,
    )

    # 4) Join the primary blocks for a compact export
    features = (
        blocks["basic_fx"]
        .join(blocks["cross"], how="outer")
        .join(blocks["pca"], how="left")
        .join(blocks["dcc"], how="left")
        .join(blocks["regime"], how="left")
    )

    # 5) Persist to CSV (ensure directory exists)
    output_file = "feature_output/fx_features_all.csv"

    features.to_csv(output_file)
    print(f"[OK] Wrote features to: {output_file}")


if __name__ == "__main__":
    # Example direct run:
    #   python fx_feature_engining.py -i "input_data/USD_EUR_daily_full.csv"
    main()
