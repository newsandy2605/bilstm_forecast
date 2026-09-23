import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TSLA Bi-LSTM Stock Forecast",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# PROJECT FILES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

EDA_IMAGE = BASE_DIR / "tsla_eda.png"
CORR_IMAGE = BASE_DIR / "tsla_corr.png"
RESULTS_IMAGE = BASE_DIR / "tsla_bilstm_results.png"
MODEL_FILE = BASE_DIR / "best_tsla_bilstm.keras"


# ============================================================
# HEADER
# ============================================================

st.title("📈 Tesla (TSLA) Stock Price Forecasting")

st.markdown(
    """
    ### Bidirectional LSTM with Technical Indicators

    An end-to-end machine learning project for forecasting Tesla's
    historical closing price using market data and technical indicators.
    """
)

st.caption(
    "Python • TensorFlow/Keras • yFinance • Scikit-learn • Streamlit"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("TSLA Bi-LSTM")

st.sidebar.markdown(
    """
    **Model configuration**

    - Ticker: TSLA
    - Lookback: 90 trading days
    - Features: 8
    - Train: 75%
    - Validation: 10%
    - Test: 15%
    - Architecture: Bidirectional LSTM
    - Loss: Huber
    """
)

st.sidebar.divider()

st.sidebar.info(
    """
    This dashboard presents the results of the trained model.
    It is an educational machine-learning project and is not
    financial advice.
    """
)


# ============================================================
# MODEL STATUS
# ============================================================

st.subheader("Model Status")

if MODEL_FILE.exists():

    st.success(
        "✅ Trained Bi-LSTM model found: "
        "`best_tsla_bilstm.keras`"
    )

else:

    st.warning(
        "⚠️ `best_tsla_bilstm.keras` was not found yet. "
        "Finish running the training script and place the generated "
        "model file in the same folder as this app."
    )


# ============================================================
# METRICS
# ============================================================

# Your original training script prints these metrics but does not
# currently save them to a JSON file. Therefore we don't invent them.
#
# Once you have the actual values from the training output, you can
# put them into the dictionary below.

metrics = {
    "MAE": None,
    "RMSE": None,
    "MAPE": None,
    "R²": None,
}


st.subheader("Model Evaluation")

m1, m2, m3, m4 = st.columns(4)

if metrics["MAE"] is not None:
    m1.metric("MAE", f"${metrics['MAE']:,.2f}")
else:
    m1.metric("MAE", "See training output")

if metrics["RMSE"] is not None:
    m2.metric("RMSE", f"${metrics['RMSE']:,.2f}")
else:
    m2.metric("RMSE", "See training output")

if metrics["MAPE"] is not None:
    m3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
else:
    m3.metric("MAPE", "See training output")

if metrics["R²"] is not None:
    m4.metric("R²", f"{metrics['R²']:.4f}")
else:
    m4.metric("R²", "See training output")


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Exploratory Analysis",
        "🧠 Model",
        "🎯 Forecast Results",
        "📁 Project Information",
    ]
)


# ============================================================
# TAB 1 — EDA
# ============================================================

with tab1:

    st.header("Exploratory Data Analysis")

    st.markdown(
        """
        The analysis includes Tesla's historical closing price,
        moving averages, Bollinger Bands, daily trading volume,
        RSI and feature correlations.
        """
    )

    if EDA_IMAGE.exists():

        st.image(
            str(EDA_IMAGE),
            caption="TSLA Exploratory Data Analysis",
            use_container_width=True,
        )

    else:

        st.error(
            "`tsla_eda.png` was not found. "
            "Run the original training script first."
        )

    st.divider()

    st.subheader("Feature Correlation")

    if CORR_IMAGE.exists():

        st.image(
            str(CORR_IMAGE),
            caption="TSLA Feature Correlation Heatmap",
            use_container_width=True,
        )

    else:

        st.error(
            "`tsla_corr.png` was not found."
        )


# ============================================================
# TAB 2 — MODEL
# ============================================================

with tab2:

    st.header("Bidirectional LSTM Architecture")

    st.markdown(
        """
        ### Input Features

        The model uses eight input features:

        - Open
        - High
        - Low
        - Close
        - Volume
        - MA_20
        - MA_50
        - RSI_14

        ### Sequence

        Each prediction uses a **90-trading-day sliding window**.

        ### Architecture

        ```
        Input
          │
          ▼
        Bidirectional LSTM — 128 units
          │
          ▼
        Batch Normalization
          │
          ▼
        Dropout — 30%
          │
          ▼
        Bidirectional LSTM — 64 units
          │
          ▼
        Batch Normalization
          │
          ▼
        Dropout — 30%
          │
          ▼
        Dense — 64 neurons
          │
          ▼
        Dropout
          │
          ▼
        Dense — 1
          │
          ▼
        Predicted Close Price
        ```

        ### Training

        The original implementation uses:

        - Adam optimizer
        - Huber loss
        - EarlyStopping
        - ReduceLROnPlateau
        - ModelCheckpoint
        """
    )

    st.divider()

    st.subheader("Training Configuration")

    config_data = pd.DataFrame(
        {
            "Parameter": [
                "Ticker",
                "Historical Start",
                "Historical End",
                "Lookback",
                "Train Split",
                "Validation Split",
                "Test Split",
                "LSTM Units",
                "Dropout",
                "Learning Rate",
                "Epochs",
                "Batch Size",
            ],
            "Value": [
                "TSLA",
                "2015-01-01",
                "2024-12-31",
                "90 trading days",
                "75%",
                "10%",
                "15%",
                "[128, 64]",
                "0.30",
                "0.001",
                "60",
                "32",
            ],
        }
    )

    st.dataframe(
        config_data,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# TAB 3 — RESULTS
# ============================================================

with tab3:

    st.header("Model Results")

    st.markdown(
        """
        The following visualization is generated by the original
        forecasting pipeline and contains the training curves,
        complete price history and the held-out test predictions.
        """
    )

    if RESULTS_IMAGE.exists():

        st.image(
            str(RESULTS_IMAGE),
            caption="Bi-LSTM Training and Test Results",
            use_container_width=True,
        )

    else:

        st.warning(
            "`tsla_bilstm_results.png` has not been generated yet. "
            "It will be created by the original training script after "
            "training and evaluation."
        )

    st.divider()

    st.subheader("What the Results Show")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            **Training / Validation**

            The first panels show how the training and validation
            loss/RMSE changed across epochs.
            """
        )

    with col2:

        st.markdown(
            """
            **Test Set**

            The lower panels compare actual TSLA closing prices
            with the model's predictions over the held-out period.
            """
        )


# ============================================================
# TAB 4 — PROJECT INFORMATION
# ============================================================

with tab4:

    st.header("About This Project")

    st.markdown(
        """
        ## Tesla Stock Price Forecasting using Bidirectional LSTM

        This project applies deep learning to historical Tesla stock
        price data.

        ### Data

        Historical TSLA market data is retrieved using Yahoo Finance.

        ### Feature Engineering

        Technical indicators include:

        - 20-day moving average
        - 50-day moving average
        - 14-period RSI
        - Bollinger Bands for exploratory analysis

        ### Preprocessing

        A chronological train/validation/test split is used to avoid
        randomly mixing future observations into the past.

        MinMaxScaler is fitted using the training data before the
        transformation is applied.

        ### Model

        The forecasting model uses two Bidirectional LSTM blocks
        followed by dense layers.

        ### Evaluation

        The original pipeline evaluates predictions using:

        - MAE
        - RMSE
        - MAPE
        - R²
        """
    )

    st.divider()

    st.subheader("Project Files")

    files = [
        ("tsla_bilstm_forecast.py", "Original ML/training pipeline"),
        ("requirements.txt", "Python dependencies"),
        ("app.py", "Streamlit web application"),
        ("tsla_eda.png", "Exploratory analysis visualization"),
        ("tsla_corr.png", "Feature correlation heatmap"),
        ("tsla_bilstm_results.png", "Model results visualization"),
        ("best_tsla_bilstm.keras", "Saved trained Keras model"),
    ]

    for filename, description in files:

        exists = (BASE_DIR / filename).exists()

        if exists:
            st.write(f"✅ **{filename}** — {description}")
        else:
            st.write(f"⚠️ **{filename}** — {description} — not found")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TSLA Bi-LSTM Stock Forecasting • "
    "Built with Python, TensorFlow/Keras and Streamlit"
)