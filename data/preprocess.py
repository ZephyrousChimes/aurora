import pandas as pd
import numpy as np
from pathlib import Path


def drop_header_if_na(df: pd.DataFrame) -> pd.DataFrame:
    """Drop first row if it contains any NaN values."""
    if df.iloc[0].isnull().sum() > 0:
        return df.iloc[1:]
    return df

def correct_column_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype({
        'Open': 'float64',
        'High': 'float64',
        'Low': 'float64',
        'Close': 'float64',
        'Volume': 'int64',
        'ticker': 'string'
    })
    return df


def parse_and_sort_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'Datetime' column to datetime and sort the dataframe."""
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df = df[df['Datetime'].notna()]
        df = df.sort_values('Datetime')
    return df

def add_log_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add a log return column if 'Close' exists."""
    if 'Close' in df.columns:
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna(subset=['log_return'])
    return df

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run full preprocessing pipeline on a single DataFrame."""
    df = drop_header_if_na(df)
    df = correct_column_types(df)
    df = parse_and_sort_datetime(df)
    df = add_log_return(df)
    return df


def preprocess_and_save(raw_dir='data/raw', output_dir='data/preprocessed'):
    raw_path = Path(raw_dir)
    pre_path = Path(output_dir)
    pre_path.mkdir(parents=True, exist_ok=True)

    for file in raw_path.glob("*.csv"):
        try:
            df = pd.read_csv(file)
            print(df)
            df = preprocess_dataframe(df)
            df.to_csv(pre_path / file.name, index=False)
            print(f"[INFO] Preprocessed: {file.name}")
        except Exception as e:
            print(f"[ERROR] Failed to preprocess {file.name}: {e}")
            return
