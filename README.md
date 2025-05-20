# Aurora

**Aurora** is a transformer-based pipeline for multi-asset financial forecasting and portfolio construction. It combines a custom Informer-style deep learning model with Black-Litterman optimization to deliver probabilistic forecasts and dynamically rebalanced portfolios.

---

## Highlights

* ✨ **Custom Informer-based Transformer** for long-range forecasting of price and variance.
* 📊 **Black-Litterman Portfolio Optimization** integrating model-implied views and equilibrium returns.
* ⚙️ **Modular CLI pipeline** for downloading data, preprocessing, and model training.
* 📈 Designed for **multi-asset workflows** over historical data from `yfinance`.

---

## Setup

Install dependencies using [Poetry](https://python-poetry.org/):

```bash
poetry install
```

---

## CLI Usage

Aurora exposes a unified CLI through `run.py` for handling the full modeling pipeline.

### 📥 Download Ticker Data

```bash
poetry run python run.py download \
    --tickers AAPL MSFT GOOG \
    --interval 1d \
    --period 10y \
    --output_dir data/raw
```

Downloads raw OHLCV data from Yahoo Finance.

---

### 🧼 Preprocess

```bash
poetry run python run.py preprocess \
    --raw_dir data/raw \
    --output_dir data/processed
```

Cleans, aligns, and prepares the raw data for training.

---

### 🧠 Train Model

```bash
poetry run python run.py train
```

Trains Aurora's Informer-based architecture using parameters from a YAML config.

> Model forecasts both expected returns and their variance over future windows.

---

## Methodology

1. **Forecasting**:

   * Custom Informer-style transformer trained on multi-asset sequences.
   * Predicts future returns and volatility jointly.

2. **Portfolio Construction**:

   * Forecasts are treated as **views** in the Black-Litterman framework.
   * Outputs optimal asset weights accounting for confidence and prior.

---

## License

Aurora is released under the **Apache License 2.0**.

---

## Contribution

Issues, discussions, and pull requests are welcome. For substantial contributions, please open an issue first to discuss the scope and direction.
