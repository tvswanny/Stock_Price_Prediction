# Stock_Price_Prediction
This program attempts to predict the next day's closing price for the S&P500 stock index.

This program uses historical price data (Open, High, Low, Close), the Adjusted Closing Price (adjusted for stock splits and dividends), and trading volume to predict the next day's closing price for the S&P500 stock index ("SPY").  Stock data is downloaded using the Yahoo Finance (yfinance) library.

The model can be tested and visualized best using the stock_prediction.ipynb JupyterHub notebook file.  To train the model, the stock_prediction.py file is used so that a GPU may be utilized to speed training.  This file allows for adjusting parameters and hyperparameters, institutes early stopping, and saves the model when optimally trained to the file stock_prediction.h5.

Once the model is trained and saved, datasets can be selected from the near 30 years of SPY data and tested using the model-test.ipynb Jupyter notebook file.  Accuracy will be calculated, results graphed, and percent profitable trades for the test period selected shown.

This model as currently constructed does not predict profitable trades overall and will need better optimization of parameters and hyperparameters if it to successfully predict daily closing prices.  Additional inputs such as technical indicators, other financial inputs such as interest rates and economic growth, and measurements of public sentiment and optimism may also be beneficial.
