import streamlit as st
import matplotlib.pyplot as plt
from stock_utils import (
    SECTOR_INDEX_MAP, SEQ_LEN, fetch_fundamentals, fetch_macro_data, get_device,
    symbols, fetch_stock, FEATURES_SEQ, FEATURES_STATIC,
    add_features, MODEL_NAME, MODELS_DIR
)
from sklearn.preprocessing import MinMaxScaler

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- MODEL LOADER ----
@st.cache_resource
def load_my_model(path):
    logger.info("started importing")
    from keras.models import load_model
    logger.info("importing done now")
    logger.info(f"Loading model from {path}...")
    model = load_model(path, compile=False)
    return model

def predict_stock(symbol, model):
    sym, df = fetch_stock(symbol)
    if df is None:
        return None

    fundamentals = fetch_fundamentals(sym)
    macro = fetch_macro_data()
    sector_index = SECTOR_INDEX_MAP.get(symbol, "^NSEI")
    df_features = add_features(df, fundamentals=fundamentals, macro=macro,
                               sector=sector_index, drop_date=True)

    df_features.rename(columns={f"{sector_index}_INDEX": "^NSEI_INDEX"}, inplace=True)

    recent_data = df_features.tail(200)
    scaler = MinMaxScaler()
    scaled_recent = scaler.fit_transform(recent_data)

    col_index_map = {c: i for i, c in enumerate(df_features.columns)}
    seq_indices = [col_index_map[c] for c in FEATURES_SEQ]
    static_indices = [col_index_map[c] for c in FEATURES_STATIC]

    X_seq = scaled_recent[-SEQ_LEN:, seq_indices].reshape(1, SEQ_LEN, len(seq_indices))
    X_static = scaled_recent[-1, static_indices].reshape(1, -1)

    pred_scaled = model.predict({"seq_input": X_seq, "static_input": X_static}, verbose=0)[0][0]

    last_row = scaled_recent[-1].copy()
    close_index_in_features_seq = FEATURES_SEQ.index("Close")
    last_row[close_index_in_features_seq] = pred_scaled
    pred_actual = scaler.inverse_transform([last_row])[0, close_index_in_features_seq]

    return {
        "symbol": symbol,
        "last_close": float(df_features["Close"].iloc[-1]),
        "predicted_close": float(pred_actual),
        "sector_index": sector_index
    }


# ---- STREAMLIT APP ----
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction (LSTM + Hybrid Features)")

# Sidebar
st.sidebar.header("Settings")
symbol = st.sidebar.selectbox("Choose Stock Symbol", symbols, index=symbols.index("TCS.NS"))

# ---- MODEL LOAD STATE ----
if "model" not in st.session_state:
    with st.spinner("Loading ML model... (this may take ~30s on first run)"):
        st.session_state.model = load_my_model(MODELS_DIR + MODEL_NAME)
    st.success("✅ Model loaded successfully!")

# Disable Predict button until model is ready
predict_btn = st.sidebar.button("Predict", disabled="model" not in st.session_state)

if  predict_btn:
    with st.spinner("Fetching data & predicting..."):
        import tensorflow
        device = get_device()
        with tensorflow.device(device):
            result = predict_stock(symbol, st.session_state.model)

    if result is None:
        st.error(f"No data available for {symbol}")
    else:
        st.success("Prediction complete!")

        st.subheader(f"Prediction for {result['symbol']}")
        st.metric(label="Last Close Price", value=f"₹{result['last_close']:.2f}")
        st.metric(label="Predicted Next Close Price", value=f"₹{result['predicted_close']:.2f}")

        sym, df_hist = fetch_stock(symbol)
        if df_hist is not None:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_hist["Date"], df_hist["Close"], label="Historical Close Price", color="blue")
            ax.axhline(result["predicted_close"], color="red", linestyle="--", label="Predicted Close")
            ax.set_title(f"{symbol} - Close Price History")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (INR)")
            ax.legend()
            st.pyplot(fig)
