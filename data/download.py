import yfinance as yf
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os

from utils import tqdm_context

def fetch_data(ticker, interval='1m', period='1d'):
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    df['ticker'] = ticker
    return df.reset_index()

def download_and_save(tickers, interval='1m', period='1d', output_dir=f"{os.path.dirname(__file__)}/raw"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with tqdm_context():
        for ticker in tqdm(tickers, desc="Downloading data"):
            df = fetch_data(ticker, interval=interval, period=period)
            file_path = output_path / f"{ticker}_{interval}_{period}.csv"
            df.to_csv(file_path, index=False)
