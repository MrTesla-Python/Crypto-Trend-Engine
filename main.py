"""
main.py

This script performs backtesting and analysis of quantitative trading strategies on cryptocurrency data.
It loads OHLCV data for a set of Binance perpetual futures, computes features, runs two strategies
(Bollinger Bands and Moving Average), combines them into a portfolio, and visualizes cumulative returns.
The code uses asynchronous data loading, pandas for data manipulation, and matplotlib for plotting.

Dependencies:
- quantpylib (custom library for Binance API and simulation)
- strategies.py (must define Bollinger and MAverage classes)
- Python libraries: asyncio, numpy, pandas, matplotlib, statsmodels, scipy

Author: Tristan Darnell
Date: 2025-07-19
"""

# Add Kelly Position Sizing
# Add more strategies potentially

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

def save_pickle(path, obj):
    """Save a Python object to a file using pickle."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    """Load a Python object from a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def compute_features(df, n):
    """
    Compute rolling volatility and normalized returns features for a DataFrame.
    Args:
        df: DataFrame with 'close' price column.
        n: Window size for rolling calculations.
    Returns:
        DataFrame with features 'x' (normalized n-period return) and 'y' (normalized next-period return).
    """
    df['logret_1'] = np.log(df['close'] / df['close'].shift(1))
    df['vol_n'] = df['logret_1'].rolling(window=n).std()
    df['x'] = np.log(df['close'] / df['close'].shift(n)) / df['vol_n']
    df['y'] = df['logret_1'].shift(-1) / df['vol_n']
    return df[['x','y']].dropna()

def regression_features(dfs, n=25):
    """
    Aggregate features for regression from multiple DataFrames (one per ticker).
    Args:
        dfs: Dict of {ticker: DataFrame}
        n: Window size for feature calculation.
    Returns:
        Concatenated DataFrame of features for all tickers.
    """
    observations = [] 
    for ticker, df in dfs.items():
        df_features = compute_features(df, n)
        df_features['ticker'] = ticker
        observations.append(df_features)
    return pd.concat(observations, ignore_index=True)

def plot(res, df):
    """
    Plot regression results with confidence intervals.
    Args:
        res: Fitted statsmodels regression result.
        df: DataFrame with features.
    """
    pred_ols = res.get_prediction()
    iv_l = pred_ols.summary_frame()['obs_ci_lower']
    iv_u = pred_ols.summary_frame()['obs_ci_upper']
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(df['x'], df['y'], 'o', label='data')
    ax.plot(df['x'], res.fittedvalues, 'r--', label='OLS')
    ax.plot(df['x'], iv_l, 'r--')
    ax.plot(df['x'], iv_u, 'r--')
    ax.legend(loc='best')
    plt.show()

# Define backtest period
start = datetime(2019, 9, 5)
end = datetime(2025, 3, 19)

async def main():
    """
    Main asynchronous function to:
    - Load or fetch OHLCV data for selected Binance perpetual futures.
    - Run Bollinger Bands and Moving Average strategies.
    - Combine strategies into a portfolio.
    - Plot cumulative returns for each strategy and the portfolio.
    """
    bin = Binance()
    # Fetch fresh data from Binance and save to pickle
    # exchange_info = await bin.exchange_info()
    # pprint(exchange_info['symbols'])
    # tickers = []
    # for symbol in exchange_info['symbols']:
    #     if symbol['status'] == 'TRADING' and symbol['contractType'] == 'PERPETUAL':
    #         tickers.append(symbol['symbol'])
    # tickers = tickers[:50]  # Limit to first 30 tickers
    # ohlcvs = await asyncio.gather(*[
    #     bin.get_trade_bars(
    #         ticker=ticker, 
    #         start=start, 
    #         end=end, 
    #         granularity='d', 
    #         granularity_multiplier=1
    #     ) for ticker in tickers
    # ])
    # save_pickle(
    #     path='data.pickle',
    #     obj=(tickers, ohlcvs)
    # )

    # Load previously saved data
    tickers, ohlcvs = load_pickle('data.pickle')
    dfs = {ticker: df for ticker, df in zip(tickers, ohlcvs)}

    # Import strategy classes
    from strategies import Bollinger, MAverage, EMATrend, WeightedEMA

    # Run Bollinger Bands strategy
    bollinger = Bollinger(
        insts=tickers,
        dfs=dfs,
        start=start,
        end=end,
        portfolio_vol=0.30,
        execrates=0.0,
    )
    bollinger_df = bollinger.run_simulation()
    print(bollinger_df)

    # Run Moving Average strategy
    maverage = MAverage(
        insts=tickers,
        dfs=dfs,
        start=start,
        end=end,
        portfolio_vol=0.30,
        execrates=0.0,
    )
    maverage_df = maverage.run_simulation()
    print(maverage_df)
    
    # Combine strategies into a portfolio

    weighted_ema = WeightedEMA(
        insts=tickers,
        dfs=dfs,
        start=start,
        end=end,
        portfolio_vol=0.30,
        execrates=0.001,
    )
    weighted_ema_df = weighted_ema.run_simulation()

    ema = EMATrend(
        insts=tickers,
        dfs=dfs,
        start=start,
        end=end,
        portfolio_vol=0.30,
        execrates=0.001,
    )
    ema_df = ema.run_simulation()


    from quantpylib.simulator.alpha import Portfolio 
    portfolio = Portfolio(
        insts=tickers,
        dfs=dfs,
        start=start,
        end=end,
        stratdfs=[bollinger_df, maverage_df, weighted_ema_df],
        portfolio_vol=0.50,
        execrates=0.001,
    )
    portfolio_df = portfolio.run_simulation()
    print('PORTFOLIO')
    print(portfolio_df)

    # Plot cumulative returns for each strategy and the portfolio
    plt.figure(figsize=(10, 6))
    plt.plot(bollinger_df.index, bollinger_df['cum_ret'], label='bollinger')
    plt.plot(maverage_df.index, maverage_df['cum_ret'], label='maverage')
    plt.plot(portfolio_df.index, portfolio_df['cum_ret'], label='portfolio')
    plt.plot(weighted_ema_df.index, weighted_ema_df['cum_ret'], label='weighted ema')
    plt.plot(ema_df.index, ema_df['cum_ret'], label='ema')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Return Over Time')
    plt.legend()
    plt.grid()
    plt.show()
    # Uncomment below to compute annualized volatility
    # r = df.capital_ret.values
    # r = r[r != 0]
    # annual_vol = np.std(r) * np.sqrt(365)
    # print(annual_vol)

    from metrics import compute_performance
    # Compute performance metrics for each strategy and the portfolio
    bollinger_perf = compute_performance(bollinger_df)
    maverage_perf = compute_performance(maverage_df)
    portfolio_perf = compute_performance(portfolio_df)
    print('BOLLINGER PERFORMANCE')
    pprint(bollinger_perf)
    print('MAVERAGE PERFORMANCE')
    pprint(maverage_perf)
    print('PORTFOLIO PERFORMANCE')
    pprint(portfolio_perf)
    
if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
