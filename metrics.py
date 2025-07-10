import numpy as np

def compute_performance(df, rf=0.0, periods_per_year=252):
    excess = df['capital_ret'] - rf
    mean_excess = np.mean(excess)
    std_excess = np.std(excess, ddof=1)

    annual_return = np.mean(df['capital_ret']) * periods_per_year
    annual_vol = np.std(df['capital_ret'], ddof=1) * np.sqrt(periods_per_year)

    sharpe_ratio = mean_excess / std_excess * np.sqrt(periods_per_year) if std_excess != 0 else np.nan

    wealth = (1 + df['capital_ret']).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    max_drawdown = drawdown.min()
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio
    }

def get_sharpe_ratio(df, rf=0.0000979913587, periods_per_year=252):
    excess = df['capital_ret'] - rf
    mean_excess = np.mean(excess)
    std_excess = np.std(excess, ddof=1)
    sharpe_ratio = mean_excess / std_excess * np.sqrt(periods_per_year) if std_excess != 0 else np.nan
    return sharpe_ratio