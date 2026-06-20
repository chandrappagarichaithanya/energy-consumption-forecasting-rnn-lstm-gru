"""
Central configuration for data, model, and training parameters.

Previously these constants were scattered as bare globals at the top of
lstm.py / seq2seq.py. Keeping them in one place makes it possible to
reuse the same values across training, evaluation, and the Streamlit app
without copy-pasting numbers that can silently drift out of sync.
"""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "kaggle_data_1h.csv"
MODELS_DIR = PROJECT_ROOT / "saved_models"

MODELS_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class LSTMConfig:
    """Hyperparameters for the single-step LSTM forecaster."""
    seq_len: int = 24            # hours of history fed into the model
    future_period: int = 6       # how many hours ahead we predict
    epochs: int = 15
    batch_size: int = 32
    lstm_units: int = 36
    test_fraction: float = 0.1
    valid_fraction: float = 0.1
    target_column: str = "Global_active_power"
    model_path: Path = MODELS_DIR / "lstm_model.keras"
    scaler_path: Path = MODELS_DIR / "lstm_scaler.pkl"


@dataclass(frozen=True)
class Seq2SeqConfig:
    """Hyperparameters for the multi-step encoder-decoder forecaster."""
    input_sequence_length: int = 72
    target_sequence_length: int = 24
    hidden_units: tuple = (30, 30)
    learning_rate: float = 0.001
    epochs: int = 15
    batch_size: int = 64
    target_column: str = "Global_active_power"
    encoder_path: Path = MODELS_DIR / "seq2seq_encoder.keras"
    decoder_path: Path = MODELS_DIR / "seq2seq_decoder.keras"
    scaler_path: Path = MODELS_DIR / "seq2seq_scaler.pkl"


LSTM_CONFIG = LSTMConfig()
SEQ2SEQ_CONFIG = Seq2SeqConfig()
