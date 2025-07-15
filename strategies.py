"""
This module defines several quantitative trading strategy classes for use in a backtesting or live trading framework.
Each strategy inherits from the Alpha base class and implements methods to compute trading signals ("alpha") based on
different technical indicators, such as Bollinger Bands, moving averages, and exponential moving averages.

Classes:
    - Bollinger: Computes alpha using Bollinger Band z-scores.
    - MAverage: Computes alpha based on the agreement of multiple simple moving average crossovers.
    - EMATrend: (Currently identical to MAverage) Computes alpha based on the agreement of multiple moving average crossovers.

Each class implements:
    - pre_compute: Calculates the alpha signal for each instrument over the date range.
    - compute_forecasts: Returns the alpha signal for eligible instruments on a given date.
    - post_compute: Cleans up alpha signals and eligibility flags after computation.
"""

import numpy as np 
import pandas as pd 

from pprint import pprint

from quantpylib.simulator.alpha import Alpha

class Bollinger(Alpha):
    """
    Bollinger Band strategy.
    Alpha is the z-score of the close price relative to its 14-period rolling mean and std.
    """
    def compute_forecasts(self, date, eligibles):
        # For each eligible instrument, return the alpha value for the given date
        forecasts = {}
        for inst in eligibles:
            forecasts[inst] = self.dfs[inst].at[date, 'alpha']
        return forecasts

    def pre_compute(self, date_range):
        # For each instrument, compute the Bollinger Band z-score and store as 'alpha'
        for inst in self.insts:
            inst_df = self.dfs[inst]
            bollinger = (inst_df['close'] - inst_df['close'].rolling(14).mean()) / inst_df['close'].rolling(14).std()
            self.dfs[inst]['alpha'] = bollinger

    def post_compute(self, date_range):
        # Forward-fill missing alpha values and update eligibility based on alpha availability
        for inst in self.insts:
            self.dfs[inst]['alpha'] = self.dfs[inst]['alpha'].ffill()
            self.dfs[inst]['eligible'] = self.dfs[inst]['eligible'] & \
                (~pd.isna(self.dfs[inst]['alpha']))

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
            trending = pd.Series(0.0, index=inst_df.index)
            for (n1, n2) in self.lookbacks:
                sma1 = inst_df.close.rolling(n1).mean()
                sma2 = inst_df.close.rolling(n2).mean()

                sig = 100 * (sma1 - sma2) / sma2
                sig = sig.clip(-20, 20)

                trending = trending.add(sig, fill_value=0)

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

    def __init__(self, lookbacks=[(2,8),(4,16),(8,32),(16,64)], **kwargs):
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

                sig =  100 * (ema1 - ema2) / ema2
                sig = sig.clip(-20, 20)

                trending = trending.add(sig, fill_value=0)

            self.dfs[inst]['alpha'] = trending.astype(np.float64)

    def post_compute(self, date_range):
        # Forward-fill missing alpha values and update eligibility based on alpha availability
        for inst in self.insts:
            self.dfs[inst]['alpha'] = self.dfs[inst]['alpha'].ffill()
            self.dfs[inst]['eligible'] = self.dfs[inst]['eligible'] & \
                (~pd.isna(self.dfs[inst]['alpha']))
            
from weight_engine import get_weights_ema
class WeightedEMA(Alpha):
    """
    Weighted Exponential Moving Average strategy.
    Computes a weighted sum of multiple EMA crossovers to generate alpha signals.
    """

    def __init__(self, lookbacks=[(2,8),(4,16),(8,32),(16,64)], weights=[0.16260140581192825, 0.32305618900082717, 0.29861334689963404, 0.21572905828761046], **kwargs):
        super().__init__(**kwargs)
        self.lookbacks = lookbacks
        if weights is None:
            weights = get_weights_ema()

        self.weights = weights / np.sum(weights)

        if len(weights) != len(lookbacks):
            raise ValueError(
                f"Expected {len(lookbacks)} weights, but got {len(weights)}"
            )
        self.weights = weights
        print("⚙️ Using weights  :", self.weights)

    def compute_forecasts(self, date, eligibles):
        # For each eligible instrument, return the alpha value for the given date
        forecasts = {}
        for inst in eligibles:
            forecasts[inst] = self.dfs[inst].at[date, 'alpha']
        return forecasts

    def pre_compute(self, date_range):
        # For each instrument, compute weighted EMA crossovers and sum them as 'alpha'
        for inst in self.insts:
            inst_df = self.dfs[inst]
            trending = pd.Series(0.0, index=inst_df.index)
            for (n1, n2), weight in zip(self.lookbacks, self.weights):
                ema1 = inst_df.close.ewm(span=n1, adjust=False).mean()
                ema2 = inst_df.close.ewm(span=n2, adjust=False).mean()

                sig = 100 * (ema1 - ema2) / ema2
                sig = sig.clip(-20, 20)

                trending = trending.add(sig * weight, fill_value=0)

            self.dfs[inst]['alpha'] = trending.astype(np.float64)

    def post_compute(self, date_range):
        # Forward-fill missing alpha values and update eligibility based on alpha availability
        for inst in self.insts:
            self.dfs[inst]['alpha'] = self.dfs[inst]['alpha'].ffill()
            self.dfs[inst]['eligible'] = self.dfs[inst]['eligible'] & \
                (~pd.isna(self.dfs[inst]['alpha']))
