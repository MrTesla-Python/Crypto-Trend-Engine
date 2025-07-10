"""
This module provides a framework for simulating and evaluating quantitative trading strategies (alphas) on a portfolio of instruments. 
It defines a base Alpha class for running backtests, handling portfolio construction, volatility targeting, transaction costs, and position management. 
The Portfolio subclass allows for combining multiple strategy outputs into a single portfolio. 
The simulation loop processes daily data, computes forecasts, updates positions, calculates PnL, and tracks portfolio statistics.
"""

import numpy as np 
import pandas as pd 

from pprint import pprint
from collections import defaultdict

# Calculates daily PnL and returns for the portfolio
def get_pnl_stats(date, prev, portfolio_df, insts, idx, dfs):
    day_pnl = 0
    nominal_ret = 0.0
    for inst in insts:
        units = portfolio_df.at[idx - 1, f'{inst} units']
        if units != 0:
            # Calculate price change and PnL for each instrument
            delta = dfs[inst].at[date, 'close'] - dfs[inst].at[prev, 'close']
            inst_pnl = units * delta
            day_pnl += inst_pnl
            # Weighted return for each instrument
            nominal_ret += portfolio_df.at[idx - 1, f'{inst} w'] * dfs[inst].at[date, 'ret']
    # Adjust return by leverage
    capital_ret = nominal_ret * portfolio_df.at[idx - 1, 'leverage']
    # Store results in portfolio dataframe
    portfolio_df.at[idx, 'pnl'] = day_pnl
    portfolio_df.at[idx, 'capital_ret'] = capital_ret
    portfolio_df.at[idx, 'nominal_ret'] = nominal_ret
    portfolio_df.at[idx, 'capital'] = portfolio_df.at[idx - 1, 'capital'] + day_pnl
    return day_pnl, capital_ret

class Alpha:
    # Base class for alpha strategy simulation

    def __init__(self, insts, dfs, start, end, portfolio_vol=0.20, execrates=0.001, position_inertia=0.10):
        self.insts = insts  # List of instruments
        self.dfs = dfs      # Dict of DataFrames for each instrument
        self.start = start
        self.end = end
        self.portfolio_vol = portfolio_vol  # Target portfolio volatility
        self.execrates = execrates          # Execution cost rate
        self.position_inertia = position_inertia  # Threshold for position change

    def compute_forecasts(self, date, eligibles):
        # To be implemented in subclass: returns forecasts for each instrument
        pass

    def get_strat_scalar(self, target_vol, ewmas, ewstrats):
        # Calculate scaling factor for strategy to target volatility
        ann_realized_vol = np.sqrt(ewmas[-1] * 365)
        return ewstrats[-1] * target_vol / ann_realized_vol

    def pre_compute(self, date_range):
        # Optional pre-computation hook
        pass 

    def post_compute(self, date_range):
        # Optional post-computation hook
        pass

    def run_simulation(self):
        # Main simulation loop
        date_range = pd.date_range(start=self.start, end=self.end, freq='D', tz='UTC')
        self.pre_compute(date_range)

        # Prepare instrument dataframes
        for inst in self.insts:
            df = pd.DataFrame(index=date_range)
            inst_vol = (-1 + self.dfs[inst]['close'] / self.dfs[inst]['close'].shift(1)).rolling(30).std()
            self.dfs[inst] = df.join(self.dfs[inst])
            self.dfs[inst][['open', 'high', 'low', 'close', 'volume']] = \
                self.dfs[inst][['open', 'high', 'low', 'close', 'volume']].ffill().bfill()
            self.dfs[inst]['ret'] = -1 + self.dfs[inst]['close'] / self.dfs[inst]['close'].shift(1)
            self.dfs[inst]['vol'] = inst_vol
            self.dfs[inst]['vol'] = self.dfs[inst]['vol'].ffill().fillna(0)
            self.dfs[inst]['vol'] = np.where(self.dfs[inst]['vol'] < 0.005, 0.005, self.dfs[inst]['vol'])

            sampled = (self.dfs[inst]['close'] != self.dfs[inst]['close'].shift(1)).fillna(False)
            eligible = sampled.rolling(5).apply(lambda x: int(np.any(x)), raw=True).fillna(0)
            self.dfs[inst]['eligible'] = eligible.astype(np.int32)

        self.post_compute(date_range)

        # Initialize and preallocate the portfolio DataFrame
        portfolio_df = pd.DataFrame(index=date_range).reset_index()
        portfolio_df = portfolio_df.rename(columns={'index': 'datetime'})

        # Preallocate all needed columns at once using pd.concat
        static_cols = pd.DataFrame({
            'capital': [0.0] * len(portfolio_df),
            'pnl': [0.0] * len(portfolio_df),
            'capital_ret': [0.0] * len(portfolio_df),
            'nominal_ret': [0.0] * len(portfolio_df),
            'nominal': [0.0] * len(portfolio_df),
            'leverage': [0.0] * len(portfolio_df),
            'cum_ret': [0.0] * len(portfolio_df),
        })

        inst_cols = {}
        for inst in self.insts:
            inst_cols[f'{inst} units'] = [0.0] * len(portfolio_df)
            inst_cols[f'{inst} w'] = [0.0] * len(portfolio_df)

        inst_cols_df = pd.DataFrame(inst_cols)
        portfolio_df = pd.concat([portfolio_df, static_cols, inst_cols_df], axis=1)

        # Set initial capital
        portfolio_df.at[0, 'capital'] = 10_000

        # Initialize state
        self.ewmas = [0.01]
        self.ewstrats = [1]
        self.strat_scalars = []
        prev_positions = defaultdict(float)

        # Simulation loop
        for i in portfolio_df.index:
            date = portfolio_df.at[i, 'datetime']
            eligibles = [inst for inst in self.insts if self.dfs[inst].at[date, 'eligible']]

            strat_scalar = 1
            if i != 0:
                date_prev = portfolio_df.at[i - 1, 'datetime']
                strat_scalar = self.get_strat_scalar(
                    target_vol=self.portfolio_vol,
                    ewmas=self.ewmas,
                    ewstrats=self.ewstrats
                )
                pnl, capital_ret = get_pnl_stats(date, date_prev, portfolio_df, self.insts, i, self.dfs)
                self.ewmas.append(0.06 * (capital_ret ** 2) + 0.94 * self.ewmas[-1] if capital_ret != 0 else self.ewmas[-1])
                self.ewstrats.append(0.06 * strat_scalar + 0.94 * self.ewstrats[-1] if capital_ret != 0 else self.ewstrats[-1])

            self.strat_scalars.append(strat_scalar)

            forecasts = self.compute_forecasts(date, eligibles)
            for k, v in forecasts.items():
                forecasts[k] = v / self.dfs[k].at[date, 'vol']

            forecast_chips = np.sum(np.abs(list(forecasts.values())))
            costs = 0.0
            nominal_tot = 0.0
            positions = defaultdict(float)

            for inst in eligibles:
                forecast = forecasts[inst]
                scaled_forecast = forecast / forecast_chips if forecast_chips != 0 else 0
                position = strat_scalar * scaled_forecast * portfolio_df.at[i, 'capital'] / self.dfs[inst].at[date, 'close']
                change = position - prev_positions[inst]
                percent_change = np.abs(change) / np.abs(position) if position != 0 else 0.0
                if percent_change < self.position_inertia:
                    position = prev_positions[inst]
                    change = 0
                costs += abs(change) * self.dfs[inst].at[date, 'close'] * self.execrates
                portfolio_df.at[i, f'{inst} units'] = position
                positions[inst] = position
                nominal_tot += abs(position * self.dfs[inst].at[date, 'close'])

            for inst in eligibles:
                units = portfolio_df.at[i, f'{inst} units']
                nominal_inst = units * self.dfs[inst].at[date, 'close']
                inst_w = nominal_inst / nominal_tot if nominal_tot != 0 else 0
                portfolio_df.at[i, f'{inst} w'] = inst_w

            portfolio_df.at[i, 'capital'] -= costs
            portfolio_df.at[i, 'nominal'] = nominal_tot
            portfolio_df.at[i, 'leverage'] = nominal_tot / portfolio_df.at[i, 'capital'] if portfolio_df.at[i, 'capital'] != 0 else 0.0
            prev_positions = positions

        # Final cleanup
        portfolio_df = portfolio_df.copy()
        portfolio_df['cum_ret'] = (1 + portfolio_df['capital_ret']).cumprod()
        return portfolio_df.set_index('datetime', drop=True)


from collections import defaultdict
class Portfolio(Alpha):
    # Combines multiple strategies into a portfolio

    def __init__(self, stratdfs, **kwargs):
        super().__init__(**kwargs)
        self.stratdfs = stratdfs  # List of strategy DataFrames

    def compute_forecasts(self, date, eligibles):
        # Combine forecasts from multiple strategies
        forecasts = defaultdict(float)
        for inst in self.insts:
            for i in range(len(self.stratdfs)):
                forecasts[inst] += self.positions[inst].at[date, i]
        return forecasts

    def post_compute(self, date_range):
        # Precompute combined positions for each instrument and strategy
        self.positions = {} 
        for inst in self.insts:
            inst_weights = pd.DataFrame(index=date_range)
            for i in range(len(self.stratdfs)):
                strat_df = self.stratdfs[i]
                inst_weights[i] = strat_df[f'{inst} w'] * strat_df['leverage']
            self.positions[inst] = inst_weights
