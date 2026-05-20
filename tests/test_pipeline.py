import pytest
import pandas as pd
from pathlib import Path

def test_streaming_raw_data_exists():
    """Pastikan folder streaming raw data tersedia."""
    streaming_path = Path("data/raw/streaming")
    assert streaming_path.exists(), "Folder data/raw/streaming tidak ditemukan!"

def test_streaming_processed_data_exists():
    """Pastikan folder streaming processed data tersedia."""
    streaming_path = Path("data/processed/streaming")
    assert streaming_path.exists(), "Folder data/processed/streaming tidak ditemukan!"

def test_processed_data_has_correct_columns():
    """Pastikan kolom hasil preprocessing sudah benar."""
    streaming_path = Path("data/processed/streaming")
    files = sorted(streaming_path.glob("*.csv"))
    assert files, "Tidak ada file processed streaming!"
    df = pd.read_csv(files[-1])
    assert "Class" in df.columns, "Kolom Class tidak ditemukan!"
    assert "amount_scaled" in df.columns, "Kolom amount_scaled tidak ditemukan!"
    assert "volume_scaled" in df.columns, "Kolom volume_scaled tidak ditemukan!"
    assert "hour" in df.columns, "Kolom hour tidak ditemukan!"

def test_no_missing_values():
    """Pastikan tidak ada missing values di data processed."""
    streaming_path = Path("data/processed/streaming")
    files = sorted(streaming_path.glob("*.csv"))
    assert files, "Tidak ada file processed streaming!"
    df = pd.read_csv(files[-1])
    assert df.isnull().sum().sum() == 0, "Ada missing values!"

def test_class_distribution():
    """Pastikan ada kelas fraud dan legitimate."""
    streaming_path = Path("data/processed/streaming")
    files = sorted(streaming_path.glob("*.csv"))
    assert files, "Tidak ada file processed streaming!"
    df = pd.read_csv(files[-1])
    assert df["Class"].nunique() == 2, "Harus ada 2 kelas (0 dan 1)!"

def test_model_exists():
    """Pastikan model sudah tersimpan."""
    assert Path("models/trained/fraud_model.pkl").exists(), \
        "Model tidak ditemukan!"

def test_ingest_script_exists():
    """Pastikan script ingestion ada."""
    assert Path("src/data/stream_generator.py").exists(), \
        "Script stream_generator.py tidak ditemukan!"

def test_preprocess_script_exists():
    """Pastikan script preprocessing ada."""
    assert Path("src/data/stream_preprocessor.py").exists(), \
        "Script stream_preprocessor.py tidak ditemukan!"

def test_stream_generator_exists():
    """Pastikan script stream generator ada."""
    assert Path("src/data/stream_generator.py").exists(), \
        "Script stream_generator.py tidak ditemukan!"
