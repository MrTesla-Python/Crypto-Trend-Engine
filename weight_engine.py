from quantpylib.simulator.alpha import Alpha
import pandas as pd
import numpy as np
from metrics import compute_performance, get_sharpe_ratio


class MAverage(Alpha):
    """
    Moving Average Crossover strategy.
    Alpha is the sum of three moving average crossover signals.
    """
    def __init__(self, lookbacks=[(10,20),(15,30),(20,50)], **kwargs):
        super().__init__(**kwargs)
        self.lookbacks = lookbacks

    def compute_forecasts(self, date, eligibles):
        # For each eligible instrument, return the alpha value for the given date
        forecasts = {}
        for inst in eligibles:
            forecasts[inst] = self.dfs[inst].at[date, 'alpha']
        return forecasts

    def pre_compute(self, date_range):
        # For each instrument, compute three moving average crossovers and sum them as 'alpha'
        for inst in self.insts:
            inst_df = self.dfs[inst]
            trending_0 = 100 * (inst_df.close.rolling(10).mean() - inst_df.close.rolling(20).mean()) / inst_df.close.rolling(20).mean()
            trending_0 = trending_0.clip(-20, 20)
            trending_1 = 100 * (inst_df.close.rolling(15).mean() - inst_df.close.rolling(30).mean()) / inst_df.close.rolling(30).mean()
            trending_1 = trending_1.clip(-20, 20)
            trending_2 = 100 * (inst_df.close.rolling(20).mean() - inst_df.close.rolling(50).mean()) / inst_df.close.rolling(50).mean()
            trending_2 = trending_2.clip(-20, 20)

            trending = trending_0 + trending_1 + trending_2
            trending = trending.astype(np.float64)
            trending[0:50] = np.nan  # Set initial periods to NaN due to insufficient data
            self.dfs[inst]['alpha'] = trending

    def post_compute(self, date_range):
        # Forward-fill missing alpha values and update eligibility based on alpha availability
        for inst in self.insts:
            self.dfs[inst]['alpha'] = self.dfs[inst]['alpha'].ffill()
            self.dfs[inst]['eligible'] = self.dfs[inst]['eligible'] & \
                (~pd.isna(self.dfs[inst]['alpha']))

class EMATrend(Alpha):
    """
    Exponential Moving Average Trend strategy.
    (Currently identical to MAverage; can be modified to use EMA instead of SMA.)
    """

    def __init__(self, lookbacks, **kwargs):
        super().__init__(**kwargs)
        self.lookbacks = lookbacks

    def compute_forecasts(self, date, eligibles):
        # For each eligible instrument, return the alpha value for the given date
        forecasts = {}
        for inst in eligibles:
            forecasts[inst] = self.dfs[inst].at[date, 'alpha']
        return forecasts

    def pre_compute(self, date_range):
        # For each instrument, compute three moving average crossovers and sum them as 'alpha'
        for inst in self.insts:
            inst_df = self.dfs[inst]
            trending = pd.Series(0.0, index=inst_df.index)
            for n1, n2 in self.lookbacks:
                ema1 = inst_df.close.ewm(span=n1, adjust=False).mean()
                ema2 = inst_df.close.ewm(span=n2, adjust=False).mean()

                sig = 100 * (ema1 - ema2) / ema2
                sig = sig.clip(-20, 20)

                trending = trending.add(sig, fill_value=0)

            self.dfs[inst]['alpha'] = trending.astype(np.float64)

    def post_compute(self, date_range):
        # Forward-fill missing alpha values and update eligibility based on alpha availability
        for inst in self.insts:
            self.dfs[inst]['alpha'] = self.dfs[inst]['alpha'].ffill()
            self.dfs[inst]['eligible'] = self.dfs[inst]['eligible'] & \
                (~pd.isna(self.dfs[inst]['alpha']))
            
            
import pickle
from datetime import datetime
def load_pickle(path):
    """Load a Python object from a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_weights_ema():
    start = datetime(2019, 9, 5)
    end = datetime(2025, 3, 19)
    tickers, ohlcvs = load_pickle('data.pickle')
    dfs = {ticker: df for ticker, df in zip(tickers, ohlcvs)}


    sharpes = []
    lookbacks = [(2,8),(4,16),(8,32),(16,64)]
    for lookback in lookbacks:
        ema = EMATrend(
        insts=tickers,
        dfs=dfs,
        start=start,
        end=end,
        portfolio_vol=0.30,
        execrates=0.001,
        lookbacks=[lookback]
    )
        ema_df = ema.run_simulation()
        sharpe = get_sharpe_ratio(ema_df)
        sharpes.append(sharpe)
    sharpes = np.array(sharpes)
    weights = sharpes / sharpes.sum()
    return weights.tolist()


def get_weights_ma():
    start = datetime(2019, 9, 5)
    end = datetime(2025, 3, 19)
    tickers, ohlcvs = load_pickle('data.pickle')
    dfs = {ticker: df for ticker, df in zip(tickers, ohlcvs)}


    sharpes = []
    lookbacks = [(10,20),(15,30),(20,50)]
    for lookback in lookbacks:
        ma = MAverage(
        insts=tickers,
        dfs=dfs,
        start=start,
        end=end,
        portfolio_vol=0.30,
        execrates=0.001,
        lookbacks=[lookback]
    )
        ma_df = ma.run_simulation()
        sharpe = get_sharpe_ratio(ma_df)
        sharpes.append(sharpe)
    sharpes = np.array(sharpes)
    weights = sharpes / sharpes.sum()
    return weights.tolist()

