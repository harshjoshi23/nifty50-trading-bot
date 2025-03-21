# NIFTY-50 Trading Bot

This project builds a trading bot using NIFTY-50 stock data (2000–2021) from Kaggle. It uses daily stock prices for 50 stocks to predict prices with an LSTM model and test trading strategies.

## Project Structure

- **data/**: Stock data files (e.g., `ADANIPORTS.csv`, `NIFTY50_all.csv`) and EDA outputs (e.g., `eda_ADANIPORTS.csv`).
- **models/**: Saved LSTM models (e.g., `lstm_ADANIPORTS.pth`).
- **src/**:
  - **data/**: `dataDownload.py` (downloads data), `preprocess.py` (prepares data for LSTM).
  - **eda/**: `eda.py`, `eda.ipynb`, `detailed_Eda.ipynb` (data analysis scripts).
  - **modeling/**: `lstm_model.py` (LSTM model training).
  - **strategies/**: `trading_strategies.py` (trading logic).
  - **trading_bot/**: `trading_bot.py` (bot implementation).
- **README.md**: This file.
- **requirements.txt**: List of Python libraries.

## Dataset Description

Each stock CSV (e.g., `ADANIPORTS.csv`) and `NIFTY50_all.csv` includes:
- `Date`: Trading date (2000–2021).
- `Symbol`: Stock ticker (e.g., ADANIPORTS).
- `Series`: Security type (e.g., EQ for equity).
- `Prev Close`: Last day’s close price.
- `Open`: Opening price.
- `High`: Day’s highest price.
- `Low`: Day’s lowest price.
- `Last`: Last traded price.
- `Close`: Closing price.
- `VWAP`: Volume Weighted Average Price.
- `Volume`: Shares traded.
- `Turnover`: Value in rupees.
- `Trades`: Number of trades (from 2011).
- `Deliverable Volume`: Shares delivered (from 2007).
- `%Deliverble`: Percentage delivered.

## What’s Done

- Analyzed data for 10 stocks to find trends and patterns.
- Trained an LSTM model to predict prices:
  - Uses 60 days of past data.
  - Model has two LSTM layers (50 units each) and one output layer.
  - Trained for 100 epochs with Adam optimizer and MSE loss.
  - Saved to `models/` (e.g., `lstm_ADANIPORTS.pth`).

## How to Run

1. **Setup**:
   - Activate environment: `source env_name/bin/activate`.
   - Install libraries: `pip install -r requirements.txt`.

2. **Get Data**:
   - Go to `src/`: `cd src`.
   - Run: `python3 data/dataDownload.py` (downloads data to `data/`).

3. **Train Model**:
   - Run: `python3 modeling/lstm_model.py` (trains LSTM and saves it).

## Dependencies

- Python 3.10
- Libraries: pandas, numpy, matplotlib, seaborn, statsmodels, torch, sklearn

## Next Step

- Build trading strategies in `trading_strategies.py` using LSTM predictions.
