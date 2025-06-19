# Markov Model.py
# -------------------------------------------------------------
# Detects market regimes for a stock using a Hidden Markov Model
# with customizable number of states and extreme regime quantiles.
# -------------------------------------------------------------

# === USER CONFIGURATION ===
TICKER           = "SPY"           # Yahoo Finance ticker
START_DATE       = "2022-01-01"     # Analysis start date (YYYY-MM-DD)
END_DATE         = None              # End date (YYYY-MM-DD) or None for today
PLOT             = True              # Show exploratory plots
SAVE_PDF         = True              # Save all plots to PDF
PDF_PATH         = "risk_regimes.pdf"  # Output PDF file

# Number of HMM regimes and quantile thresholds
N_STATES         = 4                 # Number of hidden regimes
SEED             = 42                # RNG seed for reproducibility

# Quantile thresholds for defining extreme regimes (0 < q < 1)
EXTREME_BEAR_Q   = 0.05              # States with mean return <= 20th percentile are extreme_bear
EXTREME_BULL_Q   = 0.95              # States with mean return >= 80th percentile are extreme_bull

# Feature windows / toggles
VOL_Z_WINDOW     = 30                # Days for volume z-score
VOL_WINDOW       = 20                # Days for realized volatility (rolling stdev)
USE_VIX          = True              # Include ^VIX daily return
VIX_TICKER       = "^VIX"           # External volatility index ticker
USE_SPREAD       = True              # Include bid-ask spread proxy
SPREAD_METHOD    = "high_low"       # Only (High-Low)/Close currently
USE_SENTIMENT    = False             # Include sentiment score feature
SENTIMENT_PATH   = "sentiment.csv"  # CSV w/ Date, sentiment columns

# -------------------------------------------------------------
# Dependencies: pip install yfinance hmmlearn pandas numpy matplotlib
# If USE_SENTIMENT == True: pip install vaderSentiment
# -------------------------------------------------------------
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# -------------------------------------------------------------
# 1. Data download helper
# -------------------------------------------------------------
def _download_yf(ticker: str, start: str, end: str = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data found for {ticker} between {start} and {end}.")
    return df

# -------------------------------------------------------------
# 2. Feature engineering
# -------------------------------------------------------------
def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    feat_df = pd.DataFrame(index=df.index)
    feat_df["log_ret"] = np.log(df["Close"]).diff()
    vol_mu = df["Volume"].rolling(VOL_Z_WINDOW).mean()
    vol_sd = df["Volume"].rolling(VOL_Z_WINDOW).std()
    feat_df["vol_z"] = (df["Volume"] - vol_mu) / vol_sd
    feat_df["real_vol"] = feat_df["log_ret"].rolling(VOL_WINDOW).std() * np.sqrt(252)
    if USE_SPREAD and SPREAD_METHOD == "high_low":
        feat_df["spread"] = (df["High"] - df["Low"]) / df["Close"]
    if USE_VIX:
        vdf = _download_yf(VIX_TICKER, START_DATE, END_DATE)
        feat_df["vix_ret"] = np.log(vdf["Close"]).diff().reindex(df.index)
    if USE_SENTIMENT:
        sent = pd.read_csv(SENTIMENT_PATH, parse_dates=["Date"], index_col="Date")
        sent = sent.reindex(df.index).ffill()
        for col in sent.columns:
            feat_df[col] = sent[col]
    return feat_df.dropna()

# -------------------------------------------------------------
# 3. HMM helpers
# -------------------------------------------------------------
def _fit_hmm(X: np.ndarray, k: int, seed: int) -> GaussianHMM:
    model = GaussianHMM(n_components=k, covariance_type="full", n_iter=1000, random_state=seed)
    model.fit(X)
    return model

# -------------------------------------------------------------
# 3b. State-to-Regime Mapping with Quantile Thresholds
# -------------------------------------------------------------
def _map_states_to_regimes(hmm: GaussianHMM) -> dict:
    """Map each HMM state to a regime label using mean-return quantiles."""
    mu = hmm.means_[:, 0]
    q_low = np.quantile(mu, EXTREME_BEAR_Q)
    q_high = np.quantile(mu, EXTREME_BULL_Q)
    mapping = {}
    for state, m in enumerate(mu):
        if m <= q_low:
            label = 'extreme_bear'
        elif m >= q_high:
            label = 'extreme_bull'
        else:
            # mid-range states
            if N_STATES == 2:
                label = 'bull'
            elif N_STATES == 3:
                label = 'neutral'
            else:
                median = np.median(mu)
                label = 'bear' if m < median else 'bull'
        mapping[state] = label
    return mapping

# -------------------------------------------------------------
# 4. Classification pipeline
# -------------------------------------------------------------
def classify_risk_regimes() -> pd.DataFrame:
    raw = _download_yf(TICKER, START_DATE, END_DATE)
    feat_df = _feature_engineering(raw)
    X = feat_df.values
    hmm = _fit_hmm(X, N_STATES, SEED)
    mapping = _map_states_to_regimes(hmm)
    states = hmm.predict(X)
    regimes = np.vectorize(mapping.get)(states)
    out = raw.loc[feat_df.index].copy()
    out[feat_df.columns] = feat_df
    out['regime'] = regimes
    out['regime_num'] = pd.Categorical(regimes, categories=set(mapping.values()), ordered=True).codes
    if PLOT or SAVE_PDF:
        with PdfPages(PDF_PATH) as pdf:
            _plot_regimes(out, feat_df.columns, pdf)
    return out

# -------------------------------------------------------------
# 5. Plotting (with PDF saving)
# -------------------------------------------------------------
def _plot_regimes(df: pd.DataFrame, feat_cols: list, pdf: PdfPages = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['Close'], label='Close')
    for regime in df['regime'].unique():
        mask = df['regime'] == regime
        ax.fill_between(df.index, df['Close'].min(), df['Close'].max(),
                        where=mask, alpha=0.2, label=regime)
    ax.set_title(f"{TICKER}: Price with Regime Shading")
    ax.legend()
    if pdf: pdf.savefig(fig)
    if PLOT: plt.show()
    plt.close(fig)
    for x, y in itertools.combinations(feat_cols, 2):
        fig, ax = plt.subplots(figsize=(6, 4))
        for regime in df['regime'].unique():
            mask = df['regime'] == regime
            ax.scatter(df.loc[mask, x], df.loc[mask, y], alpha=0.6, label=regime)
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{x} vs {y}")
        ax.legend(); plt.tight_layout()
        if pdf: pdf.savefig(fig)
        if PLOT: plt.show()
        plt.close(fig)

# -------------------------------------------------------------
# 6. Main
# -------------------------------------------------------------
if __name__ == "__main__":
    df = classify_risk_regimes()
    print(df.tail()[['Close', 'regime']])
