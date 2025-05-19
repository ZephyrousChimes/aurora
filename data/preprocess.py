import pandas as pd
from pathlib import Path

def preprocess_and_save(raw_dir='data/raw', output_dir='data/processed'):
    raw_path = Path(raw_dir)
    pre_path = Path(output_dir)
    pre_path.mkdir(parents=True, exist_ok=True)

    for file in raw_path.glob("*.csv"):
        try:
            df = pd.read_csv(file)

            if df.iloc[0].isnull().sum() > 0:
                df = df.iloc[1:]

            df.to_csv(pre_path / file.name, index=False)
            print(f"[INFO] Preprocessed: {file.name}")
        except Exception as e:
            print(f"[ERROR] Failed to preprocess {file.name}: {e}")
