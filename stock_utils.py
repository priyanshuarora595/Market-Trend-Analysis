import os
import json
import concurrent.futures
import yfinance as yf
import pandas as pd
import numpy as np
import platform
import sys
import subprocess
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
import inspect
try:
    from IPython import get_ipython
except ImportError:
    def get_ipython():
        return None

# -----------------------------
# Parameters
# -----------------------------
SEQ_LEN = 30
EPOCHS = 20
BATCH_SIZE = 32
MAX_WORKERS = 4
DATA_HISTORY_IN_YEARS = 20
DATA_INTERVAL_IN_DAYS = 1
DATA_DIR = "data/"
JSON_DIR = "json_files/"
MODELS_DIR = "models/"
FEATURES = ["Open", "High", "Low", "Close", "Volume"]
MODEL_NAME = "nifty100_hybrid_model_v3.keras"
DATA_DICT_FILE = "data_dict.json"
SECTOR_DICT_FILE = "sector_dict.json"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------
# Training Stocks
# -----------------------------
symbols = [
    "ADANIPORTS.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS",
    "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS",
    "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "GAIL.NS", "GRASIM.NS",
    "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
    "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS",
    "TCS.NS", "TATACONSUM.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS", "DIVISLAB.NS"
]

# Mapping NSE stocks → Sector Index (Yahoo Finance symbols)
SECTOR_INDEX_MAP = {
    "TCS.NS": "^CNXIT",        # Nifty IT
    "INFY.NS": "^CNXIT",
    "WIPRO.NS": "^CNXIT",
    "TECHM.NS": "^CNXIT",

    "HDFCBANK.NS": "^NSEBANK", # Nifty Bank
    "ICICIBANK.NS": "^NSEBANK",
    "KOTAKBANK.NS": "^NSEBANK",
    "AXISBANK.NS": "^NSEBANK",
    "SBIN.NS": "^NSEBANK",

    "RELIANCE.NS": "^CNXENERGY", # Nifty Energy
    "ONGC.NS": "^CNXENERGY",
    "BPCL.NS": "^CNXENERGY",
    "POWERGRID.NS": "^CNXENERGY",
    "NTPC.NS": "^CNXENERGY",

    # default fallback → Nifty 50 itself
}

FEATURES_SEQ = [
    "Open", "High", "Low", "Close", "Volume", "Return", "MA7", "MA21",
    "STD21", "Upper_BB", "Lower_BB", "EMA", "Momentum", "RSI",
    "MACD", "MACD_Signal", "ATR"
]

FEATURES_STATIC = ["EPS", "NIFTY", "VIX", "^NSEI_INDEX"]

SECTORS = ["^CNXIT", "^NSEBANK", "^CNXENERGY", "^NSEI"]

sector_dict = {}

# ================================
# Device selection based on system
# ================================
def get_device():
    system = platform.system()

    # ✅ Google Colab usually has GPU/TPU
    if "google.colab" in str(get_ipython()):  # detect Colab
        if tf.config.list_physical_devices("GPU"):
            return "/GPU:0"
        elif tf.config.list_physical_devices("TPU"):
            return "/TPU:0"
        else:
            return "/CPU:0"

    # ✅ macOS with M1/M2 chips -> Metal backend
    if system == "Darwin":
        if tf.config.list_physical_devices("GPU"):  # Metal is exposed as GPU
            return "/CPU:0"
        return "/CPU:0"

    # ✅ Windows/Linux -> prefer GPU if available
    if system in ["Windows", "Linux"]:
        if tf.config.list_physical_devices("GPU"):
            return "/GPU:0"
        return "/CPU:0"

    # Default fallback
    return "/CPU:0"

# -----------------------------
# Data Fetcher (cached download)
# -----------------------------

def fetch_stock(sym):
    file_path = os.path.join(DATA_DIR, f"{sym}.csv")

    if os.path.exists(file_path):
        # Load existing CSV
        df_existing = pd.read_csv(file_path, parse_dates=["Date"])

        # Find the last available date
        last_date = df_existing["Date"].max()

        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # if start_date is greater than today then return existing df
        if pd.to_datetime(start_date) > pd.to_datetime("today"):
            return sym, df_existing


        # Fetch new data starting from the day after last_date
        df_new = yf.download(
            sym,
            start=(last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval=f"{DATA_INTERVAL_IN_DAYS}d"
        )

        if not df_new.empty:
            df_new = df_new[FEATURES].dropna()

            # Flatten MultiIndex if present
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)

            df_new = df_new.reset_index()

            # Append only new rows
            print(df_existing.head())
            print(df_new.head())
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["Date"]).reset_index(drop=True)

            # Save updated CSV
            df_combined.to_csv(file_path, index=False)
            return sym, df_combined

        # If no new data, return existing
        return sym, df_existing

    else:
        try:
            # No CSV yet — fetch full history
            df = yf.download(
                sym,
                period=f"{DATA_HISTORY_IN_YEARS}y",
                interval=f"{DATA_INTERVAL_IN_DAYS}d",
                threads=False
            )
            if df is None or df.empty:
                return sym, None

            df = df[FEATURES].dropna()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.to_csv(file_path, index=False)
            return sym, df
        except Exception as e:
            print(e)
            print("Failed for no csv symbol - ", sym)
            return sym, pd.DataFrame()

# -----------------------------
# Extra Data Fetchers (for India)
# -----------------------------
def fetch_macro_data():
    """Fetch Nifty 50 Index and India VIX as macro indicators"""
    # Nifty 50 index (^NSEI) and India VIX (^INDIAVIX) are available on Yahoo Finance
    nifty = yf.download("^NSEI", period=f"{DATA_HISTORY_IN_YEARS}y", interval=f"{DATA_INTERVAL_IN_DAYS}d")
    vix = yf.download("^INDIAVIX", period=f"{DATA_HISTORY_IN_YEARS}y", interval=f"{DATA_INTERVAL_IN_DAYS}d")

    # Keep only Close prices
    nifty = nifty[["Close"]].rename(columns={"Close": "NIFTY"})
    vix = vix[["Close"]].rename(columns={"Close": "VIX"})

    # Reset index to Date
    nifty = nifty.reset_index()[["Date", "NIFTY"]]
    vix = vix.reset_index()[["Date", "VIX"]]

    # Merge
    macro = pd.merge(nifty, vix, on="Date", how="outer").sort_values("Date")

    # Flatten MultiIndex if present
    if isinstance(macro.columns, pd.MultiIndex):
        macro.columns = macro.columns.get_level_values(0)

    return macro

def fetch_fundamentals(symbol):
    """Fetch quarterly EPS from Yahoo Finance fundamentals"""
    ticker = yf.Ticker(symbol)
    try:
        financials = ticker.quarterly_financials.T  # transpose → rows=quarters
        eps = financials["Net Income"] / ticker.quarterly_financials.loc["Basic Average Shares"].T
        eps = eps.reset_index()
        eps = eps.rename(columns={"index": "Date", 0: "EPS"})
        eps["Date"] = pd.to_datetime(eps["Date"])
        return eps[["Date", "EPS"]]
    except Exception:
        # fallback if data missing
        return pd.DataFrame(columns=["Date", "EPS"])


# -----------------------------
# Updated Feature Engineering
# -----------------------------
def add_features(df: pd.DataFrame, fundamentals=None, macro=None, drop_date=False, sector=None):
    df = df.copy()

    # Technical indicators
    df["Return"] = df["Close"].pct_change()
    df["MA7"] = df["Close"].rolling(7).mean()
    df["MA21"] = df["Close"].rolling(21).mean()
    df["STD21"] = df["Close"].rolling(21).std()
    df["Upper_BB"] = df["MA21"] + (df["STD21"] * 2)
    df["Lower_BB"] = df["MA21"] - (df["STD21"] * 2)
    df["EMA"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Momentum"] = df["Close"] - df["Close"].shift(5)

    # --- RSI (Relative Strength Index, 14-day) ---
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- MACD (12,26) & Signal (9) ---
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # --- ATR (Average True Range, 14-day) ---
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = high_low.to_frame("hl").join(high_close.to_frame("hc")).join(low_close.to_frame("lc")).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Merge fundamentals (EPS quarterly)
    if fundamentals is not None and not fundamentals.empty:
        df = pd.merge(df, fundamentals, on="Date", how="left")
        # Handle duplicate EPS columns if merge creates EPS_x / EPS_y
        if "EPS" in df.columns:
            df["EPS"] = df["EPS"].ffill().bfill()
        elif "EPS_x" in df.columns and "EPS_y" in df.columns:
            df["EPS"] = df["EPS_x"].combine_first(df["EPS_y"])
            df.drop(columns=["EPS_x", "EPS_y"], inplace=True)
        elif "EPS" not in df.columns:   # ensure EPS exists
            df["EPS"] = np.nan
        df["EPS"] = df["EPS"].ffill()

    # Merge macro data (Nifty + VIX)
    if macro is not None and not macro.empty:
        df = pd.merge(df, macro, on="Date", how="left")

        # List of macro columns to handle
        macro_cols = ["NIFTY", "VIX"]

        for col in macro_cols:
            col_x, col_y = f"{col}_x", f"{col}_y"

            if col in df.columns:
                df[col] = df[col].ffill().bfill()

            elif col_x in df.columns and col_y in df.columns:
                df[col] = df[col_x].combine_first(df[col_y])
                df.drop(columns=[col_x, col_y], inplace=True)

            elif col not in df.columns:
                df[col] = np.nan

        # Forward-fill all handled macro columns
        df[macro_cols] = df[macro_cols].ffill()


    # -----------------------------
    # Sector Index (optional, e.g., NIFTY IT for TCS)
    # -----------------------------
    if sector is not None:
        if sector in sector_dict:
            sector_data = sector_dict[sector]
        else:
            sector_data = yf.download(sector, period=f"{DATA_HISTORY_IN_YEARS}y", interval=f"{DATA_INTERVAL_IN_DAYS}d")[["Close"]]
            # Flatten MultiIndex if needed
            if isinstance(sector_data.columns, pd.MultiIndex):
                sector_data.columns = sector_data.columns.get_level_values(0)
            sector_data = sector_data.reset_index().rename(columns={"Close": f"{sector}_INDEX"})

            sector_dict[sector] = sector_data

        df = pd.merge(df, sector_data, on="Date", how="left")
        if f"{sector}_INDEX" in df.columns:
                df[f"{sector}_INDEX"] = df[f"{sector}_INDEX"].ffill().bfill()
        elif f"{sector}_INDEX_x" in df.columns and f"{sector}_INDEX_y" in df.columns:
            df[f"{sector}_INDEX"] = df[f"{sector}_INDEX_x"].combine_first(df[f"{sector}_INDEX_y"])
            df.drop(columns=[f"{sector}_INDEX_x", f"{sector}_INDEX_y"], inplace=True)
        elif f"{sector}_INDEX" not in df.columns:   # ensure EPS exists
            df[f"{sector}_INDEX"] = np.nan
        df[f"{sector}_INDEX"] = df[f"{sector}_INDEX"].ffill()

    # Fill missing values instead of dropping most rows
    # df = df.ffill().bfill()

    if drop_date:
        df = df.drop(columns=["Date"])

    return df.dropna()

# -----------------------------
# Sequence Creator
# -----------------------------
def create_sequences(data, seq_len=SEQ_LEN, split=0.8):
    """
    data: full feature array (scaled)
    Assumes last 4 columns = EPS, NIFTY, VIX, Sector Index (static features)
    """

    X_seq, X_static, y = [], [], []

    # Assuming the order of FEATURES_SEQ + FEATURES_STATIC in df after adding features
    # The last 4 columns are EPS, NIFTY, VIX, and Sector Index
    data_seq = data[:, :len(FEATURES_SEQ)]
    data_static = data[:, len(FEATURES_SEQ):]
    target = data[:, 3] # Assuming 'Close' is the 4th column (index 3)

    for i in range(seq_len, len(data)):
        X_seq.append(data_seq[i-seq_len:i])
        X_static.append(data_static[i])
        y.append(target[i])

    X_seq = np.array(X_seq)
    X_static = np.array(X_static)
    y = np.array(y).reshape(-1, 1)

    split_idx = int(len(X_seq) * split)
    return (X_seq[:split_idx], X_static[:split_idx], y[:split_idx]), (X_seq[split_idx:], X_static[split_idx:], y[split_idx:])

# -----------------------------
# Training Pipeline
# -----------------------------

def save_data_dict(data_dict, file_path=DATA_DICT_FILE):
    file_path = "../json_files/"+file_path
    # Convert DataFrames to a JSON-friendly format (dictionary of lists)
    json_data_dict = {}
    for symbol, df in data_dict.items():
        # Convert DataFrame to dictionary of lists, handling datetimes
        json_data_dict[symbol] = df.to_dict(orient='list')
        # Convert datetime objects to strings for JSON compatibility
        if 'Date' in json_data_dict[symbol]:
            json_data_dict[symbol]['Date'] = [str(date) for date in json_data_dict[symbol]['Date']]
    # Dump the dictionary to a JSON file
    with open(file_path, "w") as f:
        json.dump(json_data_dict, f, indent=4)
    print(f"✅ Saved updated data_dict to {file_path}")

def load_data_dict(file_path=DATA_DICT_FILE):
    file_path = "../json_files/"+file_path
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as f:
        loaded = json.load(f)
    data_dict = {}
    for sym, data in loaded.items():
        df = pd.DataFrame(data)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        data_dict[sym] = df
    print(f"✅ Loaded existing data_dict from {file_path}")
    return data_dict

# -----------------------------
# Update Data Dictionary
# -----------------------------
def update_data_dict(symbols):
    data_dict = load_data_dict()
    sector_dict = load_data_dict(SECTOR_DICT_FILE)
    macro = fetch_macro_data()  # fetch once for all stocks
    
    # Detect environment
    system = platform.system()
    running_in_colab = "google.colab" in sys.modules

    # Choose executor type
    if running_in_colab:
        print("✅ Using ProcessPoolExecutor (Colab detected)")
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(
                tqdm(executor.map(fetch_stock, symbols), total=len(symbols), desc="Fetching data")
            )
    else:
        print(f"✅ Using sequential downloading ({system} detected)")
        results = []
        for sym in symbols:
            try:
                s, df = fetch_stock(sym)
            except Exception as e:
                print(f"Error fetching {sym}: {e}")
                continue
            results.append((sym, df))

    for sym, df in results:
        if df is not None:
            # Fetch EPS for this stock
            fundamentals = fetch_fundamentals(sym)
            sector_index = SECTOR_INDEX_MAP.get(sym, "^NSEI")

            if sym in data_dict:
                df_existing = data_dict[sym]
                df_combined = pd.concat([df_existing, df], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=["Date"]).reset_index(drop=True)
                data_dict[sym] = add_features(df_combined, fundamentals=fundamentals, macro=macro, sector= sector_index)
            else:
                data_dict[sym] = add_features(df, fundamentals=fundamentals, macro=macro, sector= sector_index)

    save_data_dict(data_dict)
    if sector_dict:
        #save sector_data to sector_data.json
        save_data_dict(sector_dict, "sector_dict.json")
    return data_dict


def predict_stock(symbol, model_path=MODELS_DIR + MODEL_NAME):
    device = get_device()
    print(f"🔹 Using device: {device}")

    with tf.device(device):
        model = load_model(model_path, compile=False)
        sym, df = fetch_stock(symbol)
        if df is None:
            print(f"❌ No data for {symbol}")
            return None

        # Add features (already merges fundamentals + macro)
        fundamentals = fetch_fundamentals(sym)
        macro = fetch_macro_data()

        # Find relevant sector index
        sector_index = SECTOR_INDEX_MAP.get(symbol, "^NSEI")  # fallback to Nifty 50
        df_features = add_features(df, fundamentals=fundamentals, macro=macro, sector=sector_index, drop_date=True)

        # rename df_features sector inde column to NSEI_Index
        df_features.rename(columns={f"{sector_index}_INDEX": "^NSEI_INDEX"}, inplace=True)

        recent_data = df_features.tail(200)
        scaler = MinMaxScaler()
        scaled_recent = scaler.fit_transform(recent_data)

        # Find indices for seq and static columns
        col_index_map = {c: i for i, c in enumerate(df_features.columns)}
        seq_indices = [col_index_map[c] for c in FEATURES_SEQ]
        static_indices = [col_index_map[c] for c in FEATURES_STATIC]

        # Last SEQ_LEN for sequential
        X_seq = scaled_recent[-SEQ_LEN:, seq_indices].reshape(1, SEQ_LEN, len(seq_indices))
        # Last row for static
        X_static = scaled_recent[-1, static_indices].reshape(1, -1) # Correctly select static features

        pred_scaled = model.predict({"seq_input": X_seq, "static_input": X_static}, verbose=0)[0][0]


        # Inverse transform (for Close price)
        last_row = scaled_recent[-1].copy()
        # Assuming 'Close' is the 4th feature in FEATURES_SEQ, which is at index 3
        close_index_in_features_seq = FEATURES_SEQ.index("Close")
        last_row[close_index_in_features_seq] = pred_scaled
        pred_actual = scaler.inverse_transform([last_row])[0, close_index_in_features_seq]

        return {
            "symbol": symbol,
            "last_close": float(df_features["Close"].iloc[-1]),
            "predicted_close": float(pred_actual),
            "sector_index": sector_index
        }
