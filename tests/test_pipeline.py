import pytest
import pandas as pd
import numpy as np
from pathlib import Path

def test_raw_data_exists():
    """Pastikan dataset mentah tersedia."""
    assert Path("data/raw/creditcard.csv").exists(), \
        "Raw dataset tidak ditemukan!"

def test_processed_data_exists():
    """Pastikan dataset processed tersedia."""
    assert Path("data/processed/creditcard_processed.csv").exists(), \
        "Processed dataset tidak ditemukan!"

def test_processed_data_has_correct_columns():
    """Pastikan kolom hasil preprocessing sudah benar."""
    df = pd.read_csv("data/processed/creditcard_processed.csv")
    assert "Class" in df.columns, "Kolom Class tidak ditemukan!"
    assert "Amount" not in df.columns, "Kolom Amount seharusnya sudah dihapus!"
    assert "Time" not in df.columns, "Kolom Time seharusnya sudah dihapus!"
    assert "Amount_scaled" in df.columns, "Kolom Amount_scaled tidak ditemukan!"
    assert "Hour" in df.columns, "Kolom Hour tidak ditemukan!"

def test_no_missing_values():
    """Pastikan tidak ada missing values di data processed."""
    df = pd.read_csv("data/processed/creditcard_processed.csv")
    assert df.isnull().sum().sum() == 0, "Ada missing values di data processed!"

def test_class_distribution():
    """Pastikan ada kelas fraud dan legitimate."""
    df = pd.read_csv("data/processed/creditcard_processed.csv")
    assert df["Class"].nunique() == 2, "Harus ada 2 kelas (0 dan 1)!"

def test_model_exists():
    """Pastikan model sudah tersimpan."""
    assert Path("models/trained/fraud_model.pkl").exists(), \
        "Model tidak ditemukan!"

def test_ingest_script_exists():
    """Pastikan script ingestion ada."""
    assert Path("src/ingest_data.py").exists(), \
        "Script ingest_data.py tidak ditemukan!"

def test_preprocess_script_exists():
    """Pastikan script preprocessing ada."""
    assert Path("src/preprocess.py").exists(), \
        "Script preprocess.py tidak ditemukan!"