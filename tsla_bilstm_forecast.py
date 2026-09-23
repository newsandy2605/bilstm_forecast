# =============================================================================
#  Tesla Inc. (TSLA) Stock Price Forecasting — Bidirectional LSTM
# =============================================================================
#  Key differences from the original Microsoft LSTM project
#  ─────────────────────────────────────────────────────────
#  1. Dataset      : Tesla (TSLA) pulled live via yfinance — no CSV needed
#  2. Features     : 8 inputs (OHLCV + MA_20 + MA_50 + RSI_14) vs. Close only
#  3. Scaler       : MinMaxScaler [0,1] — correct for bounded price data
#                    (StandardScaler was inappropriate for financial series)
#  4. Architecture : Bidirectional LSTM + BatchNormalization (Functional API)
#                    vs. vanilla Sequential LSTM
#  5. Loss         : Huber — robust to the large price spikes in TSLA
#                    vs. raw MAE
#  6. Split        : 75 / 10 / 15  train/val/test  (proper held-out val set)
#                    vs. 95 / 5 train/test with no validation set
#  7. Lookback     : 90-day sliding window vs. 60 days
#  8. Callbacks    : EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
#  9. Metrics      : MAE, RMSE, MAPE, R² on real dollar prices
# 10. Code style   : Modular functions + CONFIG dict at the top
# =============================================================================


# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    "ticker"       : "TSLA",
    "start_date"   : "2015-01-01",
    "end_date"     : "2024-12-31",
    "lookback"     : 90,           # sliding-window length (trading days)
    "train_ratio"  : 0.75,
    "val_ratio"    : 0.10,         # test gets remaining 0.15
    "epochs"       : 60,
    "batch_size"   : 32,
    "lstm_units"   : [128, 64],    # units per Bidirectional block
    "dropout_rate" : 0.30,
    "learning_rate": 1e-3,
    "patience"     : 10,           # EarlyStopping patience
    "random_seed"  : 42,
}

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # must be set BEFORE importing TF

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── TensorFlow / Keras ────────────────────────────────────────────────────────
# Your environment has keras 3.15.1 installed as a *standalone* package.
# `from tensorflow import keras` would import TF's internal bundled Keras,
# which conflicts with the standalone one.  Import keras directly instead.
import keras                            # keras 3.x standalone
import tensorflow as tf                 # kept only for random seeding

warnings.filterwarnings("ignore")
np.random.seed(CONFIG["random_seed"])
tf.random.set_seed(CONFIG["random_seed"])

# Feature list and target position (must stay in sync)
FEATURES   = ["Open", "High", "Low", "Close", "Volume", "MA_20", "MA_50", "RSI_14"]
TARGET_IDX = FEATURES.index("Close")          # index 3 → the column we predict


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA ACQUISITION
# ═════════════════════════════════════════════════════════════════════════════
def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted OHLCV data from Yahoo Finance.
    `auto_adjust=True` accounts for splits and dividends automatically.
    """
    print(f"\n[INFO] Downloading {ticker}  ({start} → {end}) …")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df.index = pd.to_datetime(df.index)

    # yfinance ≥0.2 returns MultiIndex columns when >1 ticker; flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    print(f"[INFO] {len(df):,} trading days loaded.\n")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append MA-20, MA-50, RSI-14, and Bollinger Bands.
    Bollinger Bands are used only in EDA; the model receives FEATURES above.
    """
    df = df.copy()
    # yfinance 1.x sometimes returns a 1-column DataFrame instead of a Series;
    # .squeeze() collapses it to a Series so .rolling(), .diff() etc. work correctly.
    close = df["Close"].squeeze()

    # ── Moving Averages ───────────────────────────────────────────────────────
    df["MA_20"] = close.rolling(window=20).mean()
    df["MA_50"] = close.rolling(window=50).mean()

    # ── RSI-14 (Wilder's smoothed method) ────────────────────────────────────
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI_14"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # ── Bollinger Bands (20-period, 2 σ) — EDA only ──────────────────────────
    std20          = close.rolling(20).std()
    df["BB_upper"] = df["MA_20"] + 2 * std20
    df["BB_lower"] = df["MA_20"] - 2 * std20

    df.dropna(inplace=True)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 3.  EXPLORATORY DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
def plot_eda(df: pd.DataFrame, ticker: str) -> None:
    """Three-panel EDA + correlation heatmap."""
    sns.set_style("darkgrid")

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f"{ticker} – Exploratory Data Analysis", fontsize=15, fontweight="bold")

    # Panel 1 – Close price with Moving Averages and Bollinger Bands
    ax = axes[0]
    ax.plot(df.index, df["Close"],    color="#1565C0", lw=1.4, label="Close")
    ax.plot(df.index, df["MA_20"],    color="#FB8C00", lw=1.0, ls="--", label="MA-20")
    ax.plot(df.index, df["MA_50"],    color="#6A1B9A", lw=1.0, ls="--", label="MA-50")
    ax.fill_between(df.index, df["BB_lower"], df["BB_upper"],
                    alpha=0.12, color="#1565C0", label="Bollinger Band (2σ)")
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=8)
    ax.set_title("Close Price · Moving Averages · Bollinger Bands")

    # Panel 2 – Daily trading volume
    ax = axes[1]
    ax.bar(df.index, df["Volume"], color="#43A047", alpha=0.7, width=1)
    ax.set_ylabel("Volume")
    ax.set_title("Daily Trading Volume")

    # Panel 3 – RSI with overbought / oversold lines
    ax = axes[2]
    ax.plot(df.index, df["RSI_14"], color="#C62828", lw=1.0, label="RSI-14")
    ax.axhline(70, ls="--", lw=0.8, color="#D32F2F", alpha=0.7, label="Overbought (70)")
    ax.axhline(30, ls="--", lw=0.8, color="#388E3C", alpha=0.7, label="Oversold (30)")
    ax.set_ylabel("RSI")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    ax.set_title("14-Period Relative Strength Index")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("tsla_eda.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[SAVED] tsla_eda.png")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    corr_cols = ["Open", "High", "Low", "Close", "Volume", "MA_20", "MA_50", "RSI_14"]
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", square=True, linewidths=0.5)
    plt.title(f"{ticker} – Feature Correlation Heatmap", fontsize=13, pad=14)
    plt.tight_layout()
    plt.savefig("tsla_corr.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[SAVED] tsla_corr.png")


# ═════════════════════════════════════════════════════════════════════════════
# 4.  DATA PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════
def split_and_scale(df: pd.DataFrame, train_r: float, val_r: float):
    """
    Chronological split → fit MinMaxScaler on training data only (no leakage)
    → transform the entire series.

    Returns
    -------
    scaled   : full scaled array, shape (N, n_features)
    scaler   : fitted MinMaxScaler (used to invert predictions later)
    n_train  : number of rows in the training split
    n_val    : number of rows in the validation split
    """
    values  = df[FEATURES].values
    n       = len(values)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values[:n_train])          # ← fit ONLY on training data
    scaled = scaler.transform(values)

    return scaled, scaler, n_train, n_val


def make_sequences(scaled: np.ndarray, n_train: int, n_val: int, lookback: int):
    """
    Build (X, y) sliding-window sequences for each split.

    Each validation and test split prepends `lookback` rows from the
    preceding split so the first sequence starts at the correct boundary
    without contaminating future data into the model inputs.

    Returns
    -------
    Six arrays: X_tr, y_tr, X_val, y_val, X_te, y_te
    """
    def _build(data: np.ndarray):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback: i])          # shape (lookback, n_features)
            y.append(data[i, TARGET_IDX])            # scaled Close price
        return np.array(X), np.array(y)

    X_tr,  y_tr  = _build(scaled[:n_train])
    X_val, y_val = _build(scaled[n_train - lookback: n_train + n_val])
    X_te,  y_te  = _build(scaled[n_train + n_val - lookback:])

    return X_tr, y_tr, X_val, y_val, X_te, y_te


# ═════════════════════════════════════════════════════════════════════════════
# 5.  MODEL ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════
def build_model(input_shape: tuple, units: list,
                dropout_rate: float, lr: float) -> keras.Model:
    """
    Bidirectional LSTM with BatchNormalization (Keras Functional API).

    Why Bidirectional?
    ──────────────────
    A standard LSTM reads the sequence left-to-right only.  A Bidirectional
    wrapper runs a forward and a backward LSTM in parallel and concatenates
    their outputs, doubling the effective hidden state and allowing the model
    to use both historical momentum and recency effects simultaneously.

    Why BatchNormalization?
    ───────────────────────
    Financial data has non-stationary statistics across regimes (e.g. pre/post
    COVID, interest-rate cycles).  Normalizing layer activations stabilises
    training and typically halves the number of epochs to convergence.

    Why Huber loss?
    ───────────────
    Tesla exhibits occasional 20-40 % single-day swings.  Huber is quadratic
    for small errors and linear for large ones, combining MSE-style precision
    with MAE-style outlier robustness.
    """
    inp = keras.Input(shape=input_shape, name="lstm_input")

    # Block 1 – broad pattern extraction
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(units[0], return_sequences=True), name="bi_lstm_1")(inp)
    x = keras.layers.BatchNormalization(name="bn_1")(x)
    x = keras.layers.Dropout(dropout_rate, name="do_1")(x)

    # Block 2 – distill to a fixed-size representation
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(units[1], return_sequences=False), name="bi_lstm_2")(x)
    x = keras.layers.BatchNormalization(name="bn_2")(x)
    x = keras.layers.Dropout(dropout_rate, name="do_2")(x)

    # Dense head
    x   = keras.layers.Dense(64, activation="relu", name="fc_1")(x)
    x   = keras.layers.Dropout(dropout_rate / 2, name="do_3")(x)
    out = keras.layers.Dense(1, name="price_output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="BiLSTM_TSLA")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="huber",
        metrics=["mae", keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    model.summary()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 6.  TRAINING
# ═════════════════════════════════════════════════════════════════════════════
def train_model(model,
                X_tr: np.ndarray, y_tr: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                cfg: dict):
    """
    Fit the model with three callbacks:
      • EarlyStopping       – stops when val_loss plateaus; restores best weights
      • ReduceLROnPlateau   – halves LR after 4 stagnant epochs (min 1e-6)
      • ModelCheckpoint     – saves the best checkpoint to disk
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="best_tsla_bilstm.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ═════════════════════════════════════════════════════════════════════════════
# 7.  EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
def inverse_close(col: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Invert scaling for the Close column only.
    We build a dummy (N × n_features) array, place the scaled Close values
    in the correct column, inverse-transform, then extract that column.
    """
    n_feat = len(FEATURES)
    dummy  = np.zeros((len(col), n_feat))
    dummy[:, TARGET_IDX] = col.ravel()
    return scaler.inverse_transform(dummy)[:, TARGET_IDX]


def evaluate_and_report(y_true_s: np.ndarray, y_pred_s: np.ndarray,
                         scaler: MinMaxScaler):
    """Compute MAE, RMSE, MAPE, R² on actual dollar prices and print."""
    y_true = inverse_close(y_true_s, scaler)
    y_pred = inverse_close(y_pred_s, scaler)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    r2   = r2_score(y_true, y_pred)

    sep = "=" * 48
    print(f"\n{sep}")
    print("  Test-Set Regression Metrics  (actual prices)")
    print(f"{sep}")
    print(f"  MAE   : ${mae:>10,.2f}   (avg dollar error)")
    print(f"  RMSE  : ${rmse:>10,.2f}   (penalises large errors more)")
    print(f"  MAPE  : {mape:>10.2f} %   (scale-free percentage error)")
    print(f"  R²    : {r2:>10.4f}   (1.0 = perfect fit)")
    print(f"{sep}\n")

    return y_true, y_pred


# ═════════════════════════════════════════════════════════════════════════════
# 8.  RESULTS VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════
def plot_results(history,
                 df: pd.DataFrame,
                 n_train: int, n_val: int, lookback: int,
                 y_true: np.ndarray, y_pred: np.ndarray,
                 ticker: str) -> None:
    """Four-panel dashboard: loss curves, RMSE curves, full timeline, zoomed test."""
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    fig.suptitle(f"{ticker} – Bidirectional LSTM Stock Forecast",
                 fontsize=15, fontweight="bold")

    # ── (a) Huber loss ───────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(history.history["loss"],     label="Train",      color="#1565C0", lw=1.4)
    ax.plot(history.history["val_loss"], label="Validation", color="#C62828", ls="--", lw=1.4)
    ax.set_title("Huber Loss per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber Loss")
    ax.legend()

    # ── (b) RMSE curves ──────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(history.history["rmse"],     label="Train RMSE", color="#2E7D32", lw=1.4)
    ax.plot(history.history["val_rmse"], label="Val RMSE",   color="#E65100", ls="--", lw=1.4)
    ax.set_title("RMSE per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (scaled)")
    ax.legend()

    # ── (c) Full price timeline with colour-coded regions ────────────────────
    dates  = df.index
    close  = df["Close"].values
    ax     = axes[1, 0]

    ax.plot(dates[:n_train], close[:n_train],
            color="#1565C0", lw=1.0, label="Train")
    ax.plot(dates[n_train: n_train + n_val], close[n_train: n_train + n_val],
            color="#FB8C00", lw=1.0, label="Validation")
    ax.plot(dates[n_train + n_val:], close[n_train + n_val:],
            color="#6A1B9A", lw=1.3, label="Test – Actual")

    test_dates = dates[n_train + n_val: n_train + n_val + len(y_pred)]
    ax.plot(test_dates, y_pred,
            color="#C62828", ls="--", lw=1.3, label="Test – Predicted")

    ax.set_title("Full Price History + Test Predictions")
    ax.set_ylabel("Close Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=8)

    # ── (d) Zoomed test window ────────────────────────────────────────────────
    ax     = axes[1, 1]
    n_show = min(len(y_true), len(y_pred), len(test_dates))
    ax.plot(test_dates[:n_show], y_true[:n_show],
            label="Actual",    color="#1565C0", lw=1.5)
    ax.plot(test_dates[:n_show], y_pred[:n_show],
            label="Predicted", color="#C62828", ls="--", lw=1.5)
    ax.fill_between(test_dates[:n_show], y_true[:n_show], y_pred[:n_show],
                    alpha=0.12, color="#7B1FA2")
    ax.set_title("Test Window: Actual vs Predicted")
    ax.set_ylabel("Close Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend()

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("tsla_bilstm_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[SAVED] tsla_bilstm_results.png")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = CONFIG

    # ── 1. Fetch raw OHLCV data ───────────────────────────────────────────────
    df_raw = fetch_stock_data(cfg["ticker"], cfg["start_date"], cfg["end_date"])
    print(df_raw.head()); print(df_raw.info()); print(df_raw.describe())

    # ── 2. Engineer technical indicators ─────────────────────────────────────
    df = add_technical_indicators(df_raw)
    print(f"\nDataset shape after feature engineering: {df.shape}  "
          f"({df.shape[0]} trading days × {df.shape[1]} columns)")

    # ── 3. Exploratory Data Analysis ─────────────────────────────────────────
    plot_eda(df, cfg["ticker"])

    # ── 4. Split and scale ────────────────────────────────────────────────────
    scaled, scaler, n_train, n_val = split_and_scale(
        df, cfg["train_ratio"], cfg["val_ratio"]
    )
    n_test = len(df) - n_train - n_val
    print(f"\nChrono split  →  train: {n_train:,}  val: {n_val:,}  test: {n_test:,} rows")

    # ── 5. Build sliding-window sequences ────────────────────────────────────
    lookback = cfg["lookback"]
    X_tr, y_tr, X_val, y_val, X_te, y_te = make_sequences(
        scaled, n_train, n_val, lookback
    )
    print(f"Sequence shapes  →  "
          f"X_train{X_tr.shape}  X_val{X_val.shape}  X_test{X_te.shape}")

    # ── 6. Build model ────────────────────────────────────────────────────────
    model = build_model(
        input_shape  = (lookback, len(FEATURES)),
        units        = cfg["lstm_units"],
        dropout_rate = cfg["dropout_rate"],
        lr           = cfg["learning_rate"],
    )

    # ── 7. Train ──────────────────────────────────────────────────────────────
    history = train_model(model, X_tr, y_tr, X_val, y_val, cfg)

    # ── 8. Predict & evaluate ─────────────────────────────────────────────────
    preds_scaled              = model.predict(X_te).ravel()
    y_true_price, y_pred_price = evaluate_and_report(y_te, preds_scaled, scaler)

    # ── 9. Visualise results ──────────────────────────────────────────────────
    plot_results(
        history, df, n_train, n_val, lookback,
        y_true_price, y_pred_price, cfg["ticker"]
    )

    print("\n[DONE]  Artefacts written:")
    print("        tsla_eda.png")
    print("        tsla_corr.png")
    print("        tsla_bilstm_results.png")
    print("        best_tsla_bilstm.keras")
