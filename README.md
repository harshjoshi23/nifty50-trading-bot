# NIFTY-50 Trading Bot

This project aims to develop an algorithmic trading bot using the NIFTY-50 Stock Market Data (2000–2021) dataset from Kaggle. The dataset contains historical daily stock prices for 50 stocks in the NIFTY-50 index, covering the period from 2000 to 2021. The bot will leverage machine learning (LSTM models) and technical analysis to predict price movements and execute trading strategies.

## Project Structure

- **data/**: Contains the NIFTY-50 dataset files (e.g., `ADANIPORTS.csv`, `NIFTY50_all.csv`).
- **src/**: Contains the scripts:
  - `download_data.py`: Script to download and organize the dataset.
  - (To be added) `eda.py`: Script for Exploratory Data Analysis.
  - (To be added) `train_model.py`: Script to train the LSTM model.
  - (To be added) `trading_bot.py`: Script to implement the trading bot logic.
- **README.md**: This file.

## Dataset Description

The dataset includes CSV files for each NIFTY-50 stock (e.g., `ADANIPORTS.csv`, `ASIANPAINT.csv`) and a combined file (`NIFTY50_all.csv`). Each file contains the following columns:
- `Date`: Date of the record (2000–2021).
- `Symbol`: Stock ticker (e.g., ADANIPORTS, AXISBANK).
- `Series`: Type of security (e.g., EQ for equity).
- `Prev Close`: Previous day's closing price.
- `Open`: Opening price.
- `High`: Highest price of the day.
- `Low`: Lowest price of the day.
- `Last`: Last traded price.
- `Close`: Closing price.
- `VWAP`: Volume Weighted Average Price.
- `Volume`: Trading volume.
- `Turnover`: Turnover in rupees.
- `Trades`: Number of trades (available from 2011 onwards).
- `Deliverable Volume`: Volume delivered (available from 2007 onwards).
- `%Deliverble`: Percentage of volume delivered.

## Project Goals

1. **Exploratory Data Analysis (EDA)**: Analyze the historical data to identify trends, seasonality, and correlations across stocks.
2. **Model Training**: Train an LSTM model to predict future stock prices using historical data and technical indicators (e.g., RSI, MACD).
3. **Trading Bot**: Implement a trading bot that uses model predictions and predefined strategies (e.g., momentum, mean reversion) to simulate trades.
4. **Real-Time Integration**: Extend the bot to fetch real-time data (via `yfinance`) and execute live trading strategies.

## How to Run

1. **Environment Setup**:
   - Activate the virtual environment: `source new_env/bin/activate`.
   - Install dependencies: `pip install -r requirements.txt`.

2. **Download the Dataset**:
   - Navigate to the `src/` directory: `cd src`.
   - Run the download script: `python3 download_data.py`.
   - This will download the dataset and move the CSV files to the `data/` directory.

## Dependencies

- Python 3.10
- Libraries: pandas, numpy, matplotlib, seaborn, statsmodels, torch, sklearn, ta, yfinance (listed in `requirements.txt`)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details (to be added).

## Next Steps

- Perform EDA on the dataset to understand stock price patterns.
- Train an LSTM model using historical data and technical indicators.
- Develop trading strategies and simulate trades using the bot.