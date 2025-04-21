import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.early_stop import no_progress_loss
from tqdm import tqdm
from datetime import timedelta

plt.style.use("dark_background")


def get_return_permutation(ohlc, start_time):
    ret = np.log(ohlc).diff().shift(-1).fillna(0)
    ret_perm = ret.copy()
    ret_perm.loc[start_time:] = ret_perm.loc[start_time:].values[
        np.random.permutation(len(ret.loc[start_time:]))
    ]

    return np.exp(ret_perm.cumsum()) * ohlc.iloc[0]


def get_signal_permutation(signal, start_time):
    signal_perm = signal.copy()
    signal_perm.loc[start_time:] = signal.loc[start_time:].values[
        np.random.permutation(len(signal.loc[start_time:]))
    ]
    return signal_perm


def profit_factor(signal, ohlc):
    r = np.log(ohlc["Close"]).diff().shift(-1)
    sig_rets = signal * r
    gains = sig_rets[sig_rets > 0].sum()
    losses = sig_rets[sig_rets < 0].abs().sum()

    if losses == 0:
        return 0  # avoid division by zero

    return gains / losses


def sharpe_ratio(signal, ohlc, periods_per_year=252):
    r = np.log(ohlc["Close"]).diff().shift(-1)
    sig_rets = signal * r
    mean = sig_rets.mean()
    std = sig_rets.std()

    if std == 0:
        return 0

    return mean / std * np.sqrt(periods_per_year)


def plot_perm_test(real_pf, perm_pfs, p_value, pth):
    pd.Series(perm_pfs).hist(color="blue", label="Permutations")
    plt.axvline(real_pf, color="red", label="Real")
    plt.xlabel("Sharpe Ratio")
    plt.title(f"MCPT. P-Value: {p_value}")
    plt.grid(False)
    plt.legend()
    plt.savefig(pth)
    plt.close()


class Backtester:
    def __init__(self, signal_func, param_space=None, significance_level=0.1):
        self.signal_func = signal_func
        self.param_space = param_space
        self.significance_level = significance_level

    def signal_permutation_test(self, signal, ohlc, start_time, n=1000):
        real_pf = sharpe_ratio(
            signal.loc[start_time:], ohlc.loc[start_time:]
        )

        perm_pfs = []
        for _ in tqdm(range(n)):
            perm_signal = get_signal_permutation(signal, start_time)
            perm_pfs.append(
                sharpe_ratio(
                    perm_signal.loc[start_time:], ohlc.loc[start_time:]
                )
            )

        p_value = np.sum(np.array(perm_pfs) >= real_pf) / n

        return real_pf, perm_pfs, p_value

    def return_permutation_test(self, ohlc, start_time, n=1000):
        real_signal = self.signal_func(ohlc, **self.params)
        real_pf = sharpe_ratio(
            real_signal.loc[start_time:], ohlc.loc[start_time:]
        )

        perm_pfs = []
        for _ in tqdm(range(n)):
            ohlc_perm = get_return_permutation(ohlc, start_time)
            perm_signal = self.signal_func(ohlc_perm, **self.params)
            perm_pfs.append(
                sharpe_ratio(
                    perm_signal.loc[start_time:], ohlc_perm.loc[start_time:]
                )
            )

        p_value = np.sum(np.array(perm_pfs) >= real_pf) / n
        return real_pf, perm_pfs, p_value

    def objective(self, ohlc, params):
        signal = self.signal_func(ohlc, **params)
        return {
            "loss": -sharpe_ratio(signal, ohlc),
            "status": STATUS_OK,
        }

    def tune(self, ohlc, max_evals=1000, patience=200):
        trials = Trials()
        self.params = fmin(
            fn=lambda x: self.objective(ohlc, x),
            space=self.param_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            show_progressbar=True,
            rstate=np.random.default_rng(42),
            early_stop_fn=no_progress_loss(patience),
        )
        print(self.params)

    def in_sample_evaluation(
        self, ohlc, plot_dir=".", seed=0
    ):

        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        np.random.seed(seed)
        signal = self.signal_func(ohlc, **self.params)
        ret = np.log(ohlc["Close"]).diff().shift(-1)
        print(
            f"Sharpe Ratio (in-sample): "
            f"{sharpe_ratio(signal, ohlc)}"
        )

        sig_ret = ret * signal
        plt.style.use("dark_background")
        ohlc["Close"].plot(color="red")
        plt.savefig(os.path.join(plot_dir, "price.png"))
        plt.close()

        sig_ret.cumsum().plot(color="red")
        plt.ylabel("Cumulative Log Return")
        plt.savefig(os.path.join(plot_dir, "log_returns_in_sample.png"))
        plt.close()

        real_pf, perm_pfs, p_value = self.return_permutation_test(
            ohlc, start_time=ohlc.index[0]
        )

        print(
            f"In-Sample Return Permutation test pval:"
            f"""{p_value} ({"PASS" if p_value<self.significance_level else "FAIL"})"""
        )

        plot_perm_test(
            real_pf,
            perm_pfs,
            p_value,
            os.path.join(plot_dir, "return_mcpt_is.png"),
        )

        real_pf, perm_pfs, p_value = self.signal_permutation_test(
            signal, ohlc, start_time=ohlc.index[0]
        )

        print(
            f"In-Sample Signal Permutation test pval:"
            f"""{p_value} ({"PASS" if p_value<self.significance_level else "FAIL"})"""
        )
        
        plot_perm_test(
            real_pf,
            perm_pfs,
            p_value,
            os.path.join(plot_dir, "signal_mcpt_is.png"),
        )

    def out_of_sample_evaluation(
        self,
        ohlc,
        start_test,
        test_window=timedelta(days=30),
        lookback_window=timedelta(days=365),
        max_evals=1000,
        patience=200,
        plot_dir=".",
        seed=0,
    ):

        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        np.random.seed(seed)
        signal = self.walkforward(
            ohlc,
            start_test,
            test_window,
            lookback_window,
            max_evals,
            patience,
        )

        print(
            f"Sharpe Ratio (out-of-sample): "
            f"{sharpe_ratio(signal.loc[start_test:], ohlc.loc[start_test:])}"
        )
        ret = np.log(ohlc["Close"]).diff().shift(-1)
        sig_ret = ret * signal
        sig_ret.cumsum().plot(color="red")
        plt.ylabel("Cumulative Log Return")
        plt.axvline(start_test, color="red", label="Test")
        plt.savefig(os.path.join(plot_dir, "log_returns_out_of_sample.png"))
        plt.close()

        real_pf, perm_pfs, p_value = self.signal_permutation_test(
            signal, ohlc.loc[start_test:], start_time=start_test
        )

        print(
            f"Out-Of-Sample Signal Permutation test pval:"
            f"""{p_value} ({"PASS" if p_value < self.significance_level else "FAIL"})"""
        )

        plot_perm_test(
            real_pf,
            perm_pfs,
            p_value,
            os.path.join(plot_dir, "signal_mcpt_oos.png"),
        )

    def walkforward(
        self,
        ohlc,
        start_test,
        test_window=timedelta(days=30),
        lookback_window=timedelta(days=365),
        max_evals=1000,
        patience=200,
    ):
        ohlc_test = ohlc.loc[start_test:]

        signal = pd.Series(np.full(len(ohlc_test), 0), index=ohlc_test.index)
        end_test = start_test + test_window
        start_train = start_test - lookback_window

        while end_test < ohlc.index.max():
            # self.tune(
            #     ohlc.loc[start_train:start_test],
            #     max_evals=max_evals,
            #     patience=patience,
            # )

            signal.loc[start_test:end_test] = self.signal_func(
                ohlc.loc[start_train:end_test], **self.params
            ).loc[start_test:end_test]

            start_train += test_window
            start_test += test_window
            end_test += test_window

        return signal
