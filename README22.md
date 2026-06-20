# Forecasting Energy Consumption with LSTM

Forecasts household power consumption (`Global_active_power`) several
hours ahead using an LSTM trained on the
[UCI/Kaggle household power consumption dataset](https://www.kaggle.com/uciml/electric-power-consumption-data-set).

A trained model is included (`saved_models/`), and a Streamlit app lets
you interactively run it against held-out test windows.

## Quickstart

```bash
pip install -r requirements.txt

# Run the interactive demo (uses the included pre-trained model)
streamlit run app/streamlit_app.py

# Re-train from scratch
python -m src.train_lstm --epochs 15 --n-splits 3

# Evaluate the saved model on the held-out test set
python -m src.evaluate_lstm
```

## Project structure

```
src/
  config.py          # all hyperparameters in one place
  data.py            # loading, smoothing, scaling, windowing
  models.py          # LSTM and Seq2Seq architecture definitions
  metrics.py         # RMSE / MAE / R² / MAPE / Theil's U
  train_lstm.py       # CLI training script (walk-forward CV, saves model+scaler)
  evaluate_lstm.py    # loads a saved model, reports metrics
app/
  streamlit_app.py    # interactive demo — pick a window, see the forecast
legacy/
  lstm/, seq2seq/     # original Colab-cell scripts, kept for reference
saved_models/
  lstm_model.keras, lstm_scaler.pkl   # a model already trained and ready to demo
dataset/
  kaggle_data_1h.csv
```

## What changed from the original version

The original repo (`legacy/`) was two Colab notebook cells pasted into
`.py` files — useful for understanding the approach, but not runnable
or deployable as-is. This refactor:

- **Fixed crash bugs**: `lstm.py` referenced an undefined `epochs`
  variable and an undefined `evaluate_theil` list — both would raise
  `NameError` before training finished.
- **Removed GPU-only / deprecated APIs**: `CuDNNLSTM` (removed in TF2)
  → standard `LSTM`; `keras.experimental.export_saved_model` (removed
  API) → `model.save()` with the modern `.keras` format.
- **Persisted the scaler**: the original never saved the `MinMaxScaler`
  used to normalize the data, so predictions couldn't be inverted back
  to real kWh values after the session ended. It's now pickled
  alongside the model.
- **Split into modules**: data loading, preprocessing, model
  definitions, training, and evaluation are now separate, testable
  functions instead of one long top-to-bottom script with global state.
- **Added a deployment surface**: `app/streamlit_app.py` is the part
  that was entirely missing before — a runnable interface that loads
  the trained model and produces a forecast a person can actually see.
- **`requirements.txt`** so the environment is reproducible outside
  Google Colab.

## Data Preprocessing

* Resampling to 1-hour bins
* Exponential smoothing (`alpha=0.2`)
* Outlier removal via standard deviation thresholding
* MinMax scaling, fit on the training split only (avoids leaking
  validation/test distribution into the scaler)

## Model

A 2-layer LSTM (default hidden size set in `src/config.py`) takes a
window of recent hourly power readings and predicts the value several
hours ahead. Walk-forward cross-validation
(`sklearn.model_selection.TimeSeriesSplit`) is used instead of a random
train/test split, since shuffling time-series data leaks future
information into training.

A Seq2Seq (GRU encoder-decoder) architecture for multi-step forecasting
is also available in `src/models.py::build_seq2seq_model`, ported from
the original `legacy/seq2seq/` notebook cell.

