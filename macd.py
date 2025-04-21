import numpy as np
import pandas as pd
import yfinance as yf
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
from datetime import datetime, timedelta
from backtester import Backtester


def macd(
    ohlc, short_window=5, long_window=15, buy_thresh=0.0, sell_thresh=-0.0
):
    signal = pd.Series(np.full(len(ohlc), 0), index=ohlc.index)
    if short_window > long_window:
        return signal

    macd = (
        ohlc["Close"].ewm(span=short_window).mean()
        - ohlc["Close"].ewm(span=long_window).mean()
    )

    signal[macd > buy_thresh] = 1
    signal[macd < - sell_thresh] = -1

    return signal


if __name__ == "__main__":
    data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
    data.columns = data.columns.levels[0]
    start_test = datetime(2023, 1, 1)
    param_space = {
        "short_window": hp.quniform("short_window", 1, 15, 1),
        "long_window": hp.quniform("long_window", 15, 30, 1),
        "buy_thresh": hp.loguniform("buy_thresh", np.log(1e-4), np.log(1e-2)),
        "sell_thresh": hp.loguniform(
            "sell_thresh", np.log(1e-4), np.log(1e-2)
        ),
    }
    
    backtester = Backtester(
        signal_func=macd,
        param_space=param_space
        )    

    backtester.tune(
        ohlc=data.loc[:start_test],
        max_evals=10000,
        patience=500,
    )

    backtester.in_sample_evaluation(
        ohlc=data.loc[:start_test], plot_dir="results/macd"
    )

    backtester.out_of_sample_evaluation(
        ohlc=data,
        start_test=start_test,
        max_evals=1000,
        patience=500,
        lookback_window=timedelta(days=3 * 365),
        test_window=timedelta(days=2 * 30),
        plot_dir="results/macd",
    )
