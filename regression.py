"""
regression.py

This script performs cross-sectional regression analysis on cryptocurrency price data.
It fetches OHLCV data for multiple Binance perpetual futures, computes normalized returns and volatility features,
and fits an OLS regression model to study the relationship between past and future normalized returns.
Results are visualized with confidence intervals, and binned/winsorized analysis is also performed.

Dependencies: numpy, pandas, matplotlib, statsmodels, scipy, quantpylib (custom), asyncio, pickle
"""

import pickle
import asyncio
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from pprint import pprint
from datetime import datetime 

from quantpylib.throttler.aiosonic import HTTPClient 
from quantpylib.wrappers.binance import Binance
from scipy.stats.mstats import winsorize

# Utility to save Python objects to disk
def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

# Utility to load Python objects from disk
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Compute normalized return and volatility features for a DataFrame
def compute_features(df, n):
    df['logret_1'] = np.log(df['close'] / df['close'].shift(1))  # 1-period log return
    df['vol_n'] = df['logret_1'].rolling(window=n).std()         # rolling volatility
    df['x'] = np.log(df['close'] / df['close'].shift(n)) / df['vol_n']  # n-period normalized return
    df['y'] = df['logret_1'].shift(-1) / df['vol_n']                    # next-period normalized return
    return df[['x', 'y']].dropna()

# Stack features from multiple tickers into a single DataFrame
def regression_features(dfs, n=25):
    'r_f / vol_n ~ r_h / vol_n'
    observations = [] 
    for ticker, df in dfs.items():
        df_features = compute_features(df, n)
        df_features['ticker'] = ticker
        observations.append(df_features)
    return pd.concat(observations, ignore_index=True)

# Plot regression fit and confidence intervals
def plot(res, df):
    pred_ols = res.get_prediction()
    iv_l = pred_ols.summary_frame()['obs_ci_lower']
    iv_u = pred_ols.summary_frame()['obs_ci_upper']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df['x'], df['y'], 'o', label='data')
    ax.plot(df['x'], res.fittedvalues, 'r--', label='OLS')
    ax.plot(df['x'], iv_l, 'r--')
    ax.plot(df['x'], iv_u, 'r--')
    ax.legend(loc='best')
    plt.show()

# Define analysis period
start = datetime(2019, 1, 19)
end = datetime(2025, 3, 19)

# Main async function to fetch data, compute features, run regression, and plot results
async def main():
    bin = Binance()
    exchange_info = await bin.exchange_info()
    pprint(exchange_info['symbols'])  # Print available symbols

    # Uncomment below to fetch and save fresh data
    # tickers = []
    # # Select only trading perpetual contracts
    # for symbol in exchange_info['symbols']:
    #     if symbol['status'] == 'TRADING' and symbol['contractType'] == 'PERPETUAL':
    #         tickers.append(symbol['symbol'])
    # tickers = tickers[:30]  # Limit to 30 tickers for speed
    # ohlcvs = await asyncio.gather(*[
    #     bin.get_trade_bars(
    #         ticker=ticker, 
    #         start=start, 
    #         end=end, 
    #         granularity='d', 
    #         granularity_multiplier=1
    #     ) for ticker in tickers
    # ])
    # print(ohlcvs)
    # save_pickle(
    #     path='data.pickle',
    #     obj=(tickers, ohlcvs)
    # )

    # Load previously saved data
    tickers, ohlcvs = load_pickle('data.pickle')
    dfs = {ticker: df for ticker, df in zip(tickers, ohlcvs)}
    stacked_df = regression_features(dfs)
    res = smf.ols('y ~ x', data=stacked_df).fit()
    print(res.summary())

    #plot(res, stacked_df)  # Uncomment to plot raw regression

    # Bin x into quantiles, winsorize, and aggregate for robust regression
    stacked_df['x_bin'] = pd.qcut(stacked_df['x'], q=200)
    stacked_df = stacked_df.drop(columns=['ticker'])
    stacked_df = stacked_df.groupby('x_bin').agg(
        lambda grp: np.mean(winsorize(grp, limits=[0.05, 0.05]))
    ).reset_index() 
    binned_model = smf.ols('y ~ x', data=stacked_df).fit()
    plot(binned_model, stacked_df)

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
