"""
Binance Futures API Wrapper

This module provides an asynchronous Python wrapper for interacting with the Binance USDT-margined Futures API.
It allows users to fetch exchange information and historical candlestick (kline) data for trading pairs, with
support for rate limiting and flexible time granularities. The wrapper is designed to be used in quantitative
trading and research workflows.

Key Features:
- Maps user-friendly time granularities to Binance's interval format.
- Fetches historical kline/candlestick data in batches, handling API limits.
- Provides exchange information endpoint.
- Handles rate limiting using AsyncRateSemaphore and aconsume_credits decorator.
"""

import pytz
import pandas as pd
import numpy as np

from quantpylib.throttler.aiosonic import HTTPClient
from quantpylib.throttler.rate_semaphore import AsyncRateSemaphore
from quantpylib.throttler.decorators import aconsume_credits

USDM_BASE_URL = "https://fapi.binance.com"

def map_to_bin_granularities(granularity, granularity_multiplier):
    """
    Map the granularity and granularity_multiplier to Binance's interval format.

    Args:
        granularity (str): Time unit ('w', 'M', 'd', 'h', 'm').
        granularity_multiplier (int): Multiplier for the time unit.

    Returns:
        tuple: (interval string for Binance, interval in seconds)
    """
    if granularity == 'w':
        return f"{granularity_multiplier}w", granularity_multiplier * 7 * 24 * 60 * 60 
    elif granularity == 'M':
        return f"{granularity_multiplier}M", granularity_multiplier * 31 * 24 * 60 * 60
    elif granularity == 'd':
        return f"{granularity_multiplier}d", granularity_multiplier * 24 * 60 * 60
    elif granularity == 'h':
        return f"{granularity_multiplier}h", granularity_multiplier * 60 * 60
    elif granularity == 'm':
        return f"{granularity_multiplier}m", granularity_multiplier * 60
    else:
        raise ValueError("Unsupported granularity. Use 'w', 'M', 'd', 'h', 'm', or 's'.")

# API endpoint definitions for Binance Futures
endpoints = {
    'kline_candlestick': {
        'endpoint': '/fapi/v1/klines',
        'method': 'GET',
    },
    'exchange_info': {
        'endpoint': '/fapi/v1/exchangeInfo',
        'method': 'GET',
    }
}

class Binance():
    """
    Asynchronous wrapper for Binance USDT-margined Futures API.
    """

    def __init__(self, key='', secret='', **kwargs):
        """
        Initialize the Binance API wrapper.

        Args:
            key (str): API key (optional, not used in this code).
            secret (str): API secret (optional, not used in this code).
        """
        self.key = key
        self.secret = secret
        self.http_client = HTTPClient(base_url=USDM_BASE_URL)
        self.rate_semaphore = AsyncRateSemaphore(6000)  # Rate limiter for API calls

    async def exchange_info(self):
        """
        Fetch exchange information (symbols, trading rules, etc.) from Binance.

        Returns:
            dict: Exchange information.
        """
        request = dict(endpoints['exchange_info'])
        return await self.http_client.request(**request)

    async def get_trade_bars(self, ticker, start, end, granularity, granularity_multiplier, kline_close=False):
        """
        Fetch historical candlestick (kline) data for a given symbol and time range.

        Args:
            ticker (str): Trading pair symbol (e.g., 'BTCUSDT').
            start (datetime): Start time (timezone-aware or naive UTC).
            end (datetime): End time (timezone-aware or naive UTC).
            granularity (str): Time unit ('w', 'M', 'd', 'h', 'm').
            granularity_multiplier (int): Multiplier for the time unit.
            kline_close (bool): If True, use kline close time as datetime index.

        Returns:
            pd.DataFrame: DataFrame with columns ['open', 'high', 'low', 'close', 'volume'] indexed by datetime.
        """
        gran_str, gran_secs = map_to_bin_granularities(granularity, granularity_multiplier)
        # Ensure start and end are timezone-aware (UTC)
        if start.tzinfo is None: start = start.replace(tzinfo=pytz.UTC)
        if end.tzinfo is None: end = end.replace(tzinfo=pytz.UTC)
        unix_start, unix_end = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
        results = []
        # Fetch data in batches (up to 1490 candles per request)
        while unix_start < unix_end:
            res = await self.kline_candlestick(
                symbol=ticker,
                interval=gran_str,
                startTime=unix_start,
                endTime=unix_start + gran_secs * 1000 * 1490,  # Fetch up to 1490 candles at a time
                limit=1500
            )
            # If no data is returned and nothing has been collected, skip ahead
            if len(res) == 0 and len(results) == 0:
                unix_start = unix_start + gran_secs * 1000
                continue
            # If no more data or stuck at same timestamp, break loop
            if len(res) == 0 or unix_start == res[-1][0]:
                break
            else:
                unix_start = res[-1][0]
            results.extend(res)
        df = pd.DataFrame(results)
        # Assign column names as per Binance API response
        df.columns = ['t', 'open', 'high', 'low', 'close', 'volume', 'T', '-', '-', '-', '-', '-']
        if df.empty:
            return pd.DataFrame()
        # Choose which timestamp to use as datetime index
        dt_col = {"T": 'datetime'} if kline_close else {"t": 'datetime'}
        df = df.rename(columns=dt_col)
        # Keep only relevant columns
        df = df.drop(columns=[col for col in list(df) if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']])
        # Convert datetime column to pandas datetime (UTC)
        df.datetime = pd.to_datetime(df.datetime, unit='ms', utc=True)
        df = df.set_index('datetime', drop=True)
        # Remove duplicate indices, keep last
        df = df[~df.index.duplicated(keep='last')]
        # Filter by requested time range
        df = df[start:end]
        # Convert price/volume columns to float
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(np.float64)
        return df

    @aconsume_credits(costs=10, refund_in=60)
    async def kline_candlestick(self, symbol, interval, startTime=None, endTime=None, limit=None):
        """
        Fetch candlestick (kline) data from Binance for a given symbol and interval.

        Args:
            symbol (str): Trading pair symbol.
            interval (str): Binance interval string (e.g., '1m', '1h').
            startTime (int): Start time in milliseconds since epoch.
            endTime (int): End time in milliseconds since epoch.
            limit (int): Maximum number of data points to fetch.

        Returns:
            list: List of kline data arrays as returned by Binance API.
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': startTime,
            'endTime': endTime,
            'limit': limit
        }
        # Remove None values from params
        params = {k: v for k, v in params.items() if v is not None}
        request = dict(endpoints['kline_candlestick'])
        request['params'] = params
        return await self.http_client.request(**request)