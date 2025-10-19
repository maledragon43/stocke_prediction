# streamlit_alpha_av.py  
import os
import time
from io import BytesIO
from typing import Optional, Dict

import pandas as pd
import requests
import streamlit as st
import zipfile  

DEFAULT_API_KEY = "ITG8KVPK6NIOSOFV"
BASE_URL = "https://www.alphavantage.co/query"

# ---------------------------
# Static choices / UI lists
# ---------------------------

SECTOR_CHOICES = [
    # Alphabetical, matches the common GICS broad sectors used in many screeners
    "Energy",
    "Materials",
    "Industrials",
    "Utilities",
    "Healthcare",
    "Financials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Information Technology",
    "Communication Services",
    "Real Estate",
]

EXCHANGE_CHOICES = [
    # US-focused venues you filter against in list.csv / LISTING_STATUS
    "NYSE",
    "AMEX",
    "BATS",
    "NASDAQ",
    "NYSE ARCA",
    "NYSE MKT",
]

# Mapping from UI interval → (Alpha Vantage function, expected top-level JSON key)

FN_BY_INTERVAL = {
    # Daily  returns the JSON object with key "Time Series (Daily)".
    "daily":   ("TIME_SERIES_DAILY",   "Time Series (Daily)"),

    # Weekly adjusted → JSON key "Weekly  Time Series"
    "weekly":  ("TIME_SERIES_WEEKLY",  "Weekly Time Series"),

    # Monthly adjusted → JSON key "Monthly Time Series"
    "monthly": ("TIME_SERIES_MONTHLY", "Monthly Time Series"),
}

# ---------------------------
# FX codes shown in the FX tab
# ---------------------------
# List of (ISO code, human-readable name). 
# - FX_CODES = [code for code, _ in FX_CURRENCIES]
# - FX_NAME_MAP = {code: name, ...}
# ...and use them to render selectboxes with friendly labels (USD (US Dollar), etc.).
FX_CURRENCIES = [
    ("USD", "US Dollar"),
    ("EUR", "Euro"),
    ("GBP", "British Pound"),
    ("JPY", "Japanese Yen"),
    ("CHF", "Swiss Franc"),
    ("CAD", "Canadian Dollar"),
    ("AUD", "Australian Dollar"),
    ("NZD", "New Zealand Dollar"),
    ("CNY", "Chinese Yuan"),
    ("HKD", "Hong Kong Dollar"),
    ("SGD", "Singapore Dollar"),
    ("KRW", "South Korean Won"),
    ("TWD", "New Taiwan Dollar"),
    ("INR", "Indian Rupee"),
    ("IDR", "Indonesian Rupiah"),
    ("MYR", "Malaysian Ringgit"),
    ("THB", "Thai Baht"),
    ("PHP", "Philippine Peso"),
    ("VND", "Vietnamese Dong"),
    ("AED", "UAE Dirham"),
    ("SAR", "Saudi Riyal"),
    ("ILS", "Israeli Shekel"),
    ("TRY", "Turkish Lira"),
    ("RUB", "Russian Ruble"),
    ("ZAR", "South African Rand"),
    ("EGP", "Egyptian Pound"),
    ("NGN", "Nigerian Naira"),
    ("KES", "Kenyan Shilling"),
    ("PLN", "Polish Zloty"),
    ("CZK", "Czech Koruna"),
    ("HUF", "Hungarian Forint"),
    ("SEK", "Swedish Krona"),
    ("NOK", "Norwegian Krone"),
    ("DKK", "Danish Krone"),
    ("RON", "Romanian Leu"),
    ("UAH", "Ukrainian Hryvnia"),
    ("GEL", "Georgian Lari"),
    ("MDL", "Moldovan Leu"),
    ("MXN", "Mexican Peso"),
    ("BRL", "Brazilian Real"),
    ("CLP", "Chilean Peso"),
    ("COP", "Colombian Peso"),
    ("PEN", "Peruvian Sol"),
    ("ARS", "Argentine Peso"),
]


def ensure_state_defaults():
    """
    Initialize Streamlit session_state keys used across tabs.

    Why this exists:
    - Streamlit re-runs the script on every interaction. session_state lets you
      persist values (loaded universe, user selections, built ZIP bytes, etc.)
      so buttons/selections don't reset unexpectedly.

    Keys:
      - have_universe: whether the symbol universe (after filters) is ready.
      - candidates: list[str] of symbols matching Exchange+Sector.
      - listings_df: the full listings DataFrame (list.csv or LISTING_STATUS).
      - selected_symbols: symbols the user picked (must equal 'limit').
      - ms_symbols_seeded: guards one-time seeding of defaults into multiselect.
      - zip_bytes: raw bytes of the in-memory ZIP to download.
      - ts_interval: "daily"/"weekly"/"monthly" used when fetching series.
      - last_filters: tuple(exchange, sector, state) to detect changes and reset.
      - name_by_symbol: mapping "AAPL" → "Apple Inc." for pretty labels.
    """
    st.session_state.setdefault("have_universe", False)
    st.session_state.setdefault("candidates", [])
    st.session_state.setdefault("listings_df", None)
    st.session_state.setdefault("selected_symbols", [])
    st.session_state.setdefault("ms_symbols_seeded", False)
    st.session_state.setdefault("zip_bytes", None)
    st.session_state.setdefault("ts_interval", "daily")
    st.session_state.setdefault("last_filters", None)
    st.session_state.setdefault("name_by_symbol", {})  # SYM → Name mapping


@st.cache_data
def load_csv_cached(path: str) -> pd.DataFrame:
    """
    Read a CSV from disk and cache the resulting DataFrame.

    Streamlit cache:
      - @st.cache_data memoizes *by argument value* and data contents.
      - If 'path' doesn't change and the file content hash is the same,
        subsequent calls return instantly from cache (no I/O).

    Args:
      path: filesystem path to a CSV file (e.g., "list.csv", "sector_map.csv").

    Returns:
      A pandas DataFrame loaded from the CSV.
    """
    return pd.read_csv(path)


def get_api_key() -> str:
    """
    Resolve the Alpha Vantage API key with a user-first fallback strategy.

    Order of precedence:
      1) st.session_state["api_key_override"] (what the user typed in the sidebar)
      2) Environment variable ALPHAVANTAGE_API_KEY (set via OS or .env)
      3) DEFAULT_API_KEY (hardcoded fallback for quick demos)

    Returns:
      The chosen API key as a non-empty string (may be DEFAULT_API_KEY).
    """
    override = st.session_state.get("api_key_override", "").strip()
    if override:
        return override
    # DEFAULT_API_KEY and BASE_URL are defined elsewhere in your file
    return os.environ.get("ALPHAVANTAGE_API_KEY", DEFAULT_API_KEY)


def fetch_listing_status(apikey: str, state: str = "active") -> pd.DataFrame:
    """
    Fetch Alpha Vantage LISTING_STATUS (CSV) and return as a DataFrame.

    What it is:
      - A full universe of tickers and metadata (symbol, name, exchange, assetType,
        ipoDate, delistingDate, status, etc.).
      - 'state' filters by 'active' vs 'delisted' on the provider side.

    Notes:
      - Alpha Vantage returns raw CSV bytes for this endpoint.
      - We use BytesIO to read the in-memory bytes directly with pandas (no temp file).
      - This call is network-bound; you may want caching at a higher level.

    Args:
      apikey: Your Alpha Vantage API key.
      state:  "active" (default) or "delisted".

    Returns:
      DataFrame with the provider's listing schema. Common columns include:
      ['symbol', 'name', 'exchange', 'assetType', 'ipoDate', 'delistingDate', 'status']
    """
    params = {
        "function": "LISTING_STATUS",  # AV function name
        "state": state,                # 'active' or 'delisted'
        "apikey": apikey,
    }

    # Perform the HTTP GET with a sane timeout; raise_for_status ensures 4xx/5xx
    # become exceptions rather than silent failures.
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()

    # Parse CSV directly from bytes; avoids writing to disk.
    return pd.read_csv(BytesIO(resp.content))

def filter_by_exchange(df: pd.DataFrame, exchange: str) -> pd.DataFrame:
    """
    Return a slice of `df` matching the requested exchange.

    Strategy:
    - Hard-require an exchange string; if empty/missing, return an empty frame.
    - If the 'exchange' column isn't present, also return empty (can't filter).
    - First try an exact case-insensitive match (best quality).
    - If none found, fallback to a substring/contains match (more permissive).
    """
    if not exchange:
        # No exchange provided -> return an empty frame with same columns
        return df.iloc[0:0]
    if "exchange" not in df.columns:
        # Defensive: upstream CSV might not have 'exchange'
        return df.iloc[0:0]

    ex = exchange.strip().lower()

    # Exact, case-insensitive match (fast path)
    exact = df[df["exchange"].astype(str).str.lower() == ex]
    if not exact.empty:
        return exact

    # Fallback: partial match (e.g., user typed 'ny' and you want 'NYSE' rows)
    return df[df["exchange"].astype(str).str.lower().str.contains(ex)]


def fetch_overview(apikey: str, symbol: str):
    """
    Call Alpha Vantage 'OVERVIEW' to fetch company fundamentals/metadata.

    Returns:
      - dict with fields like MarketCapitalization, PERatio, Sector, etc., or
      - None if the provider returns a rate-limit 'Note'/'Information' message,
        an error, or a non-dict payload.

    Notes:
      - 'OVERVIEW' is JSON. Rate-limits often come back as {'Note': ...}.
    """
    params = {"function": "OVERVIEW", "symbol": symbol, "apikey": apikey}
    r = requests.get(BASE_URL, params=params, timeout=60)

    try:
        data = r.json()
        # Reject non-dict, empty dict, or informational/rate-limit payloads
        if not isinstance(data, dict) or not data or ("Note" in data) or ("Information" in data):
            return None
        return data
    except Exception:
        # If JSON parsing fails (rare but possible), normalize to None
        return None


def fetch_series(
    apikey: str,
    symbol: str,
    interval: str,
    max_retries: int = 3,
    throttle_sec: float = 12.0
) -> Dict:
    """
    Fetch historical price series from Alpha Vantage as JSON and return the raw dict.

    Parameters:
      apikey       : Alpha Vantage key.
      symbol       : Ticker symbol to fetch.
      interval     : 'daily'|'weekly'|'monthly' (mapped via FN_BY_INTERVAL).
      max_retries  : How many retries if rate-limited or transient error.
      throttle_sec : Sleep between retries (helps avoid rate limits).

    Contract:
      - Uses FN_BY_INTERVAL[interval] → (function_name, expected_series_key).
      - Validates that the expected top-level key exists in the JSON.
      - On repeated failure, raises RuntimeError with the last error payload.

    Pitfalls handled:
      - Rate limits / 'Note'/'Information' messages.
      - 4xx/5xx via raise_for_status.
      - Unexpected JSON layout (missing expected series key).
    """
    fn, series_key = FN_BY_INTERVAL[interval]
    params = {
        "function": fn,
        "symbol": symbol,
        "apikey": apikey,
        "outputsize": "full",   # ask for full history (provider may still cap)
        "datatype": "json",     # ensure JSON for predictable parsing
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(BASE_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()

            # AV rate-limit/notes/errors often come as dicts with these keys
            if (not isinstance(data, dict)) or (not data) or ("Note" in data) or ("Information" in data) or ("Error Message" in data):
                last_err = data  # keep entire payload for debugging
            else:
                # Ensure the time-series key we expect is present
                if series_key not in data:
                    last_err = {"error": f"Expected key '{series_key}' not found.", "keys": list(data.keys())}
                else:
                    return data  # success
        except Exception as e:
            # Network errors, JSON parse errors, etc.
            last_err = {"exception": str(e)}

        # Not done yet and we failed -> back off to reduce rate-limit hits
        if attempt < max_retries:
            time.sleep(throttle_sec)

    # If we get here, all attempts failed
    raise RuntimeError(f"Alpha Vantage request failed after {max_retries} attempts: {last_err}")


def normalize_to_df(data: Dict, interval: str) -> pd.DataFrame:
    """
    Convert Alpha Vantage time series JSON to a tidy DataFrame.

    Inputs:
      data     : Raw JSON dict from fetch_series(...)
      interval : 'daily'|'weekly'|'monthly' (only used to resolve the series key)

    Output schema (subset; depends on endpoint):
      ['date','open','high','low','close','adjusted_close','volume','dividend_amount','split_coefficient', ...]

    Notes:
      - AV uses numbered keys like '1. open', '5. adjusted close', etc.
      - Some series provide both '5. volume' and '6. volume' depending on adjusted/non-adjusted.
      - We sanitize numeric fields; if parsing fails, set to pandas NA.
    """
    _, series_key = FN_BY_INTERVAL[interval]
    ts = data[series_key]  # dict of {date_str: {field_name: value, ...}}

    rows = []
    for date_str, fields in ts.items():
        rec = {"date": pd.to_datetime(date_str)}

        # Map AV's numbered keys to consistent column names
        mapping = {
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adjusted_close",
            # AV sometimes uses these positions for 'volume' depending on series
            "5. volume": "volume",
            "6. volume": "volume",
            "7. dividend amount": "dividend_amount",
            "8. split coefficient": "split_coefficient",
        }

        for k, v in fields.items():
            key = mapping.get(k, k)  # default: keep original if unmapped
            try:
                if key in ["volume"]:
                    # Volume may be string; cast via float first to be safe, then int
                    rec[key] = int(float(v))
                elif key in ["split_coefficient"]:
                    rec[key] = float(v)
                else:
                    # Prices, adjusted_close, dividend_amount, etc.
                    rec[key] = float(v)
            except Exception:
                # If any conversion fails, store NA so downstream ops don't crash
                rec[key] = pd.NA

        rows.append(rec)

    # Build frame, sort chronologically, clean index
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Put the most useful columns first; keep any additional columns at the end
    ordered = ["date", "open", "high", "low", "close", "adjusted_close", "volume", "dividend_amount", "split_coefficient"]
    cols = [c for c in ordered if c in df.columns] + [c for c in df.columns if c not in ordered]
    return df[cols]


def fetch_realtime_options(
    apikey: str,
    symbol: str,
    require_greeks: bool = False,
    contract: str = "",
    datatype: str = "json"
) -> pd.DataFrame:
    """
    Fetch realtime options from Alpha Vantage.

    Parameters:
      apikey        : Alpha Vantage key.
      symbol        : Underlying (e.g., 'AAPL').
      require_greeks: If True, request implied vol and Greek fields (if plan supports).
      contract      : Optional specific contract filter (provider-specific semantics).
      datatype      : 'json' (default) or 'csv'.

    Returns:
      - If csv => pandas DataFrame loaded from CSV bytes.
      - If json => normalize 'option_chain'/'data' list into a DataFrame.
      - Empty DataFrame if the payload isn't recognized (e.g., rate-limit note).
    """
    params = {"function": "REALTIME_OPTIONS", "symbol": symbol, "apikey": apikey, "datatype": datatype}
    if require_greeks:
        params["require_greeks"] = "true"
    if contract:
        params["contract"] = contract

    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()

    if datatype == "csv":
        # Provider returns literal CSV content; load directly from memory
        return pd.read_csv(BytesIO(r.content))

    # JSON path: try common keys; AV varies between 'option_chain' and 'data'
    data = r.json()
    chain = data.get("option_chain") or data.get("data") or []
    if isinstance(chain, list):
        # Flatten list-of-dicts (handles nested keys like 'greeks.iv', etc.)
        return pd.json_normalize(chain)
    return pd.DataFrame()


def fetch_fx_rate(apikey: str, from_symbol: str, to_symbol: str) -> dict:
    """
    Fetch the latest FX quote for a pair (e.g., USD->EUR).

    Endpoint:
      function=CURRENCY_EXCHANGE_RATE

    Returns:
      A dict containing fields like '5. Exchange Rate', '6. Last Refreshed', etc.
      If the provider responds with a different shape (rate-limit note), you'll
      likely get an empty dict from the `.get(...)` call below.
    """
    params = {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": from_symbol,
        "to_currency": to_symbol,
        "apikey": apikey
    }
    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json().get("Realtime Currency Exchange Rate", {})


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to CSV (no index) and return raw bytes, ready for zipping
    or download_button streaming in Streamlit.
    """
    return df.to_csv(index=False).encode()


# ---------------------------
# FX label helpers used in UI
# ---------------------------

# List of ISO codes used for the selectboxes (derived from your FX_CURRENCIES tuples)
FX_CODES = [c for c, _ in FX_CURRENCIES]

# Mapping code -> full name for pretty labels ('USD' -> 'US Dollar')
FX_NAME_MAP = {c: n for c, n in FX_CURRENCIES}


def fx_label(code: str) -> str:
    """
    Build a friendly label for a currency code.

    'USD' -> 'USD (US Dollar)'
    If the name isn't known, just return the code.
    """
    name = FX_NAME_MAP.get(code, "")
    return f"{code} ({name})" if name else code


# ---------------------------
# Streamlit page scaffolding
# ---------------------------

# Configure the page (title + wide layout for data tables)
st.set_page_config(page_title="Alpha Vantage Toolkit", layout="wide")

# Big header at the top of the app
st.title("Alpha Vantage Toolkit (Equities • Options • FX)")


# Sidebar: API key & docs
with st.sidebar:
    st.subheader("API Key")
    # Allow user to override the key at runtime; persisted in session_state["api_key_override"]
    st.text_input(
        "Alpha Vantage API Key",
        value=os.environ.get("ALPHAVANTAGE_API_KEY", DEFAULT_API_KEY),
        key="api_key_override"
    )

    # Helpful hint showing the last 4 chars of the active key
    st.write(
        "Using key ending:",
        ("***" + get_api_key()[-4:]) if get_api_key() else "(none)"
    )

    st.divider()
    # Quick link to provider docs (opens in new tab)
    st.markdown("**Docs**: https://www.alphavantage.co/documentation/")

# Main tab layout:
#  1) Equities: build a universe by Exchange+Sector, fetch OHLCV, bundle ZIP
#  2) Options: realtime chain browser / downloader
#  3) FX: historical series downloader (CSV)
tabs = st.tabs(["Equities (Exchange + Sector)", "Options", "FX"])


@st.cache_data
def load_underlyings(list_path="list.csv", state="active", exchanges=None):
    """
    Load a full underlying universe from a local CSV (Alpha Vantage LISTING_STATUS export or similar),
    normalize key columns, and apply simple filters.

    Caching:
      - @st.cache_data memoizes the DataFrame by input args and file content hash,
        so repeated calls are instant unless the CSV changes on disk.

    Args:
      list_path: Path to a CSV containing at least 'symbol', optionally 'name','exchange','status'.
      state    : If provided, keep only rows with that 'status' (e.g., 'active' or 'delisted').
      exchanges: Optional list of exchange codes (e.g., ['NASDAQ','NYSE']) to include.

    Returns:
      Deduplicated, symbol-sorted DataFrame with normalized 'symbol' case and guaranteed
      presence of 'name','exchange','status' columns (added if missing).
    """
    df = pd.read_csv(list_path)

    # --- Normalize core columns ---
    # 1) Symbols to uppercase, strip whitespace
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    # 2) Ensure 'name','exchange','status' exist so downstream code doesn't crash
    if "name" not in df.columns:
        df["name"] = ""
    if "exchange" not in df.columns:
        df["exchange"] = ""
    if "status" not in df.columns:
        df["status"] = "active"

    # --- Filters ---
    # Filter by listing status if requested (case-insensitive)
    if state:
        df = df[df["status"].astype(str).str.lower() == state.lower()]
    # Optional exchange filter (exact match on provided list)
    if exchanges:
        df = df[df["exchange"].astype(str).isin(exchanges)]

    # Remove duplicate tickers (keep first occurrence) and sort for stable UI
    df = df.drop_duplicates(subset="symbol").sort_values("symbol")
    return df


def pretty_label(sym: str, name_by_symbol: dict) -> str:
    """
    Compose a user-friendly label for selectboxes:
      'AAPL (Apple Inc.)' if a name is known, else just 'AAPL'.
    """
    nm = name_by_symbol.get(sym, "")
    return f"{sym} ({nm})" if nm else sym


# ------------------ EQUITIES ------------------
with tabs[0]:
    # Ensure all session_state keys exist (prevents KeyError on first render)
    ensure_state_defaults()

    st.subheader("Filter by BOTH Exchange and Sector")

    # --- Filters row ---
    # Four same-width columns: Exchange, Sector, Limit, Listing State
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        # Required input: exchange code (used for universe filtering by your sector_map.csv + list.csv)
        exchange = st.selectbox("Exchange (required)", EXCHANGE_CHOICES, index=0, key="exchange")
    with c2:
        # Required input: sector (drives selection from sector_map.csv)
        sector = st.selectbox("Sector (required)", SECTOR_CHOICES, index=0, key="sector")
    with c3:
        # Limit the *number of symbols* to fetch time series for (helps avoid rate limits)
        limit = st.number_input("How many equities? (≤ 5)", 1, 5, 2, 1, key="limit")
    with c4:
        # Which listing universe to pull from (matches Alpha Vantage LISTING_STATUS 'status')
        state = st.selectbox("Listing state", ["active", "delisted"], index=0, key="listing_state")

    # Guard: require both filters before enabling the fetch button
    ready = bool(exchange.strip()) and bool(sector.strip())
    if not ready:
        st.info("Please enter BOTH an Exchange and a Sector to proceed.")

    # --- Detect filter changes & invalidate universe ---
    # If any of (exchange, sector, state) changed vs previous render, we clear everything
    current_filters = (exchange.strip(), sector.strip(), state.strip())
    filters_changed = (st.session_state["last_filters"] != current_filters)

    if filters_changed:
        # Reset universe + selections so the user must reload with the new filters
        st.session_state.have_universe = False
        st.session_state.candidates = []
        st.session_state.listings_df = None
        st.session_state.ms_symbols_seeded = False
        st.session_state.selected_symbols = []
        st.session_state.zip_bytes = None
        st.session_state.name_by_symbol = {}     # Reset pretty-label map
        st.session_state["ms_symbols"] = []      # Clear multiselect selection
        st.info("Filters changed — click **Fetch Equities (Exchange + Sector)** to load new symbols.")

    # --- Fetch universe (ONE-TIME until filters change) ---
    if st.button("Fetch Equities (Exchange + Sector)", type="primary", disabled=not ready, key="btn_fetch"):
        apikey = get_api_key()  # not used here yet, but kept for parity if you switch to LIVE LISTING_STATUS

        try:
            # Your app uses local cache files (fast). You can swap to fetch_listing_status(...) if desired.
            listings = load_csv_cached("list.csv")      # Big universe (symbol,name,exchange,status,...)
            df_map   = load_csv_cached("sector_map.csv")  # Your mapping: symbol -> sector
        except Exception as e:
            # Any I/O or parse error is surfaced and stops the app flow cleanly
            st.error(f"LISTING_STATUS/MAP error: {e}")
            st.stop()

        # Filter the sector map down to the selected sector (case-insensitive)
        sec = sector.strip().lower()
        df_sector = df_map[df_map["sector"].astype(str).str.lower() == sec]

        # Unique uppercase symbols for that sector
        symbols = df_sector["symbol"].astype(str).str.strip().str.upper().unique()

        # Join the listings universe to the sector symbols (keeps listing metadata like 'name')
        listings_sector = listings[listings["symbol"].astype(str).str.strip().str.upper().isin(symbols)].copy()

        # Build the candidate list shown in the multiselect
        candidates = (
            listings_sector["symbol"]
            .astype(str).str.strip().str.upper().unique().tolist()
        )

        # --- SYM -> Name mapping for pretty labels (e.g., 'MSFT (Microsoft Corp)') ---
        # Robustly pull 'name'; if your CSV uses a different header, add fallbacks here.
        nm_series = (
            listings_sector.assign(
                _sym=listings_sector["symbol"].astype(str).str.strip().str.upper(),
                _name=listings_sector.get("name", "").astype(str).fillna("").str.strip()
            )
            .set_index("_sym")["_name"]
        )
        name_by_symbol = nm_series.to_dict()

        # Handle no matches (e.g., sector/exchange combo not present in CSV)
        if not candidates:
            st.warning("No matches for the chosen exchange + sector in mapping.")
            st.session_state.have_universe = False
            st.session_state.candidates = []
            st.session_state.listings_df = None
            st.session_state.ms_symbols_seeded = False
            st.session_state.selected_symbols = []
            st.session_state.name_by_symbol = {}
        else:
            # Persist the universe + metadata in session_state
            st.session_state.have_universe = True
            st.session_state.candidates = candidates
            st.session_state.listings_df = listings
            st.session_state.last_filters = current_filters
            st.session_state.name_by_symbol = name_by_symbol

            # Seed the multiselect ONCE with up to 'limit' tickers
            seeded = candidates[:min(int(limit), len(candidates))]
            st.session_state.selected_symbols = seeded
            st.session_state["ms_symbols"] = seeded
            st.session_state.ms_symbols_seeded = True

            st.success(f"Loaded {len(candidates)} candidate symbol(s). Scroll down to select & download.")

    # --- Selection + time series (rendered when universe is ready) ---
    if st.session_state.have_universe:
        st.markdown("### Select symbols")
        st.caption(f"Pick exactly {int(limit)} symbol(s) from the filtered list below.")

        # Local pretty-label function that pulls from session_state mapping
        def _pretty(sym: str) -> str:
            nm = st.session_state.name_by_symbol.get(sym, "")
            return f"{sym} ({nm})" if nm else sym

        # The *values* of the multiselect are still raw symbols; only the display is pretty
        st.multiselect(
            "Symbols",
            options=st.session_state.candidates,
            key="ms_symbols",
            format_func=_pretty,
            help="Choose exactly the number set in 'How many equities?'",
        )
        # Sync selection into a simpler key for downstream logic
        st.session_state.selected_symbols = list(st.session_state["ms_symbols"])

        # Time-series interval selector (ties into FN_BY_INTERVAL downstream)
        st.markdown("### Time series settings")
        st.session_state.ts_interval = st.selectbox(
            "Interval", ["daily", "weekly", "monthly"],
            index=["daily", "weekly", "monthly"].index(st.session_state.ts_interval),
            key="ts_interval_select"
        )

        # Enforce exact selection count (helps manage API limits and UI promises)
        if len(st.session_state.selected_symbols) != int(limit):
            st.error(f"Please select exactly {int(limit)} symbol(s). Currently selected: {len(st.session_state.selected_symbols)}.")
            can_build = False
        else:
            can_build = True

        # If filters changed since last universe fetch, prevent building to avoid mismatch
        if st.session_state["last_filters"] != current_filters:
            st.warning("Filters have changed — please click **Fetch Equities** to refresh the symbols.")
            can_build = False

        # Build ZIP of CSV time-series for selected symbols
        if st.button("Fetch & Build ZIP", type="primary", key="btn_build_zip", disabled=not can_build):
            apikey = get_api_key()
            zip_buffer = BytesIO()
            # Create an in-memory ZIP archive we’ll later expose via download_button
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                progress = st.progress(0.0)
                for i, sym in enumerate(st.session_state.selected_symbols, start=1):
                    try:
                        # Fetch raw JSON series for the chosen interval
                        # TIP: Ensure FN_BY_INTERVAL has correct function+series_key (adjusted weekly/monthly).
                        raw = fetch_series(apikey, sym, st.session_state.ts_interval, max_retries=3, throttle_sec=5.0)
                        # Normalize JSON into a tidy DataFrame
                        df_ts = normalize_to_df(raw, st.session_state.ts_interval)
                        # Turn it into CSV bytes to write into the ZIP
                        csv_bytes = dataframe_to_csv_bytes(df_ts)
                    except Exception as e:
                        # If a symbol fails (rate-limit, missing data, etc.), include a small "error" CSV
                        # so the user can see what happened for that ticker.
                        csv_bytes = f"error,symbol,message\n1,{sym},{str(e).replace(',', ';')}\n".encode()
                        st.warning(f"{sym}: {e}")

                    # Try to fetch a friendly company name for the filename
                    name = st.session_state.listings_df.loc[
                        st.session_state.listings_df["symbol"].astype(str).str.strip().str.upper() == sym, "name"
                    ]
                    name_str = name.iloc[0] if not name.empty else ""
                    # Sanitize filename (remove invalid characters for Windows/macOS)
                    safe_name = "".join(c for c in name_str if c not in '\\/:*?"<>|').strip()
                    fname = f"{sym}{(' (' + safe_name + ')') if safe_name else ''}.csv"

                    # Write CSV to the ZIP
                    zf.writestr(fname, csv_bytes)

                    # Update progress bar and throttle to respect API limits
                    progress.progress(i / len(st.session_state.selected_symbols))
                    time.sleep(5.0)  # extra pacing within the loop

            # Make the ZIP bytes available across reruns
            zip_buffer.seek(0)
            st.session_state.zip_bytes = zip_buffer.getvalue()
            st.success(f"Prepared {len(st.session_state.selected_symbols)} CSV file(s). See download button below.")

    # --- Download button persists across reruns ---
    # If a previous run produced a ZIP, keep offering it (until filters reset cleared it)
    if st.session_state.get("zip_bytes"):
        st.download_button(
            "Download ZIP of CSVs",
            data=st.session_state.zip_bytes,
            file_name=f"{exchange}_{sector}_{st.session_state.ts_interval}_{int(limit)}_symbols.zip",
            mime="application/zip",
            key="btn_download_zip"
        )


# ------------------ OPTIONS TAB ------------------
with tabs[1]:
    st.subheader("Realtime Options Chain (US)")

    # Attempt to pull a cleaned universe for dropdowns
    # (Change the filters or expose them as UI elements if needed.)
    try:
        under_df = load_underlyings(
            list_path="list.csv",
            state="active",  # Only actively listed underlyings
            exchanges=["NASDAQ", "NYSE", "AMEX", "NYSE ARCA", "NYSE MKT"]
        )
        # Prepare pretty labels for the selectbox
        name_by_symbol = dict(zip(under_df["symbol"], under_df["name"]))
        under_symbols = under_df["symbol"].tolist()

        # Quick text filter across symbol and name (helps when universe is large)
        q = st.text_input("Filter symbols by text (optional)", "", placeholder="type part of symbol or name…")
        if q:
            qlow = q.strip().lower()
            mask = under_df["symbol"].str.contains(qlow, case=False) | under_df["name"].str.lower().str.contains(qlow)
            view_df = under_df[mask]
            name_by_symbol = dict(zip(view_df["symbol"], view_df["name"]))
            under_symbols = view_df["symbol"].tolist()

        # Prefer default = AAPL when available; else first item (if any)
        default_idx = under_symbols.index("AAPL") if "AAPL" in under_symbols and len(under_symbols) > 0 else 0

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            under = st.selectbox(
                "Underlying Symbol",
                options=under_symbols,
                index=default_idx if under_symbols else 0,
                format_func=lambda s: pretty_label(s, name_by_symbol),
                key="opt_under_sel",
            )
        with c2:
            greeks = st.checkbox("Require Greeks & IV", value=False)
        with c3:
            st.caption(f"{len(under_symbols):,} underlyings")

    except Exception as _e:
        # If list.csv isn’t present, fall back to manual text inputs
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            under = st.text_input("Underlying Symbol", "AAPL")
        with c2:
            greeks = st.checkbox("Require Greeks & IV", value=False)
        with c3:
            st.caption("Using manual input (list.csv not found)")

    # Fetch and show options data
    if st.button("Fetch Options", key="btn_opt", type="primary"):
        apikey = get_api_key()
        with st.spinner("Fetching options..."):
            try:
                # HISTORICAL_OPTIONS returns CSV. If you want live chain snapshots,
                # consider REALTIME_OPTIONS (JSON) and transform as needed.
                params = {"function": "HISTORICAL_OPTIONS", "symbol": under, "apikey": apikey, "datatype": "csv"}
                if greeks:
                    params["require_greeks"] = "true"  # Supported on certain plans

                r = requests.get(BASE_URL, params=params, timeout=60)
                r.raise_for_status()

                # Parse returned CSV bytes into a DataFrame
                df = pd.read_csv(BytesIO(r.content))
                if df.empty:
                    st.warning("No data returned. The symbol may not be optionable on your plan or at this time.")
                else:
                    st.success(f"Returned {len(df):,} rows for {under}.")
                    st.dataframe(df, use_container_width=True)
                    # Offer a simple CSV download of the exact payload
                    st.download_button("Download CSV", df.to_csv(index=False).encode(), "options.csv", "text/csv")
            except Exception as e:
                st.error(f"Options error: {e}")


# ------------------ FX ------------------
with tabs[2]:
    st.subheader("Realtime FX Rate")

    # Two columns for the currency pair selectors
    c1, c2 = st.columns([1, 1])

    # Choose reasonable defaults (USD→EUR) if present in the FX_CODES list
    from_idx = FX_CODES.index("USD") if "USD" in FX_CODES else 0
    to_idx   = FX_CODES.index("EUR") if "EUR" in FX_CODES else 0

    with c1:
        fx_from = st.selectbox(
            "From Currency",
            options=FX_CODES,
            index=from_idx,
            format_func=fx_label,  # Renders 'USD (US Dollar)' etc.
            key="fx_from",
        )
    with c2:
        fx_to = st.selectbox(
            "To Currency",
            options=FX_CODES,
            index=to_idx,
            format_func=fx_label,
            key="fx_to",
        )

    # Optional historical series choices (for CSV fetch)
    c4, c5 = st.columns([1, 1])
    with c4:
        fx_horizon = st.selectbox("Horizon", ["Daily", "Weekly", "Monthly"], index=0, key="fx_horizon")
    with c5:
        outputsize = st.selectbox("Output size", ["compact", "full"], index=1, key="fx_outputsize")

    # Warn if the pair is identical (no meaningful rate)
    if fx_from == fx_to:
        st.warning("From and To currencies are the same. Choose different codes to get a rate.")

    # Map the chosen horizon to Alpha Vantage's function name
    fn_map = {
        "Daily": "FX_DAILY",
        "Weekly": "FX_WEEKLY",
        "Monthly": "FX_MONTHLY",
    }
    fx_func = fn_map[fx_horizon]

    # Trigger the FX series fetch
    if st.button("Get FX CSV", type="primary", disabled=(fx_from == fx_to)):
        apikey = get_api_key()
        with st.spinner(f"Fetching {fx_horizon.lower()} FX series..."):
            try:
                params = {
                    "function": fx_func,
                    "from_symbol": fx_from,   # Note: FX_* endpoints use from_symbol/to_symbol
                    "to_symbol": fx_to,
                    "apikey": apikey,
                    "datatype": "csv",        # Ask for CSV directly for easy downloading
                    "outputsize": outputsize, # compact ~100 rows, full = full history (if supported)
                }
                r = requests.get(BASE_URL, params=params, timeout=60)
                r.raise_for_status()
                content = r.content or b""

                # AV sometimes returns JSON (rate-limit notes) even when you request CSV.
                looks_json = content.strip().startswith(b"{") or content.strip().startswith(b"[")
                if looks_json:
                    # Try to parse the JSON and surface the provider's message to the user
                    msg = r.json()
                    st.error(f"Alpha Vantage response (not CSV): {msg.get('Error Message') or msg.get('Note') or str(msg)[:300]}")
                else:
                    # Parse the returned CSV bytes and preview a small slice in the UI
                    df = pd.read_csv(BytesIO(content))
                    if df.empty:
                        st.warning("No FX data returned.")
                    else:
                        st.success(f"Returned {len(df):,} rows for {fx_from}/{fx_to} ({fx_horizon}).")
                        st.dataframe(df.head(50), use_container_width=True)

                        # Offer the CSV for download exactly as returned by the provider
                        fname = f"{fx_from}_{fx_to}_{fx_horizon.lower()}_{outputsize}.csv"
                        st.download_button(
                            "Download CSV",
                            data=content,
                            file_name=fname,
                            mime="text/csv",
                            key="fx_dl_csv",
                        )
            except Exception as e:
                st.error(f"FX error: {e}")

#streamlit run streamlit_alpha_av.py



