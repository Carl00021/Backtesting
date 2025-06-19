# risk_regime_hmm.py
# -------------------------------------------------------------
# Detects market regimes for a stock using a Hidden Markov Model
# with customizable number of states and extreme regime quantiles,
# and tracks posterior probabilities over time.
# -------------------------------------------------------------

# === USER CONFIGURATION ===
TICKER           = "VSTS"           # Yahoo Finance ticker
START_DATE       = "2000-01-01"     # Analysis start date (YYYY-MM-DD)
END_DATE         = None              # End date (YYYY-MM-DD) or None for today
PLOT             = False              # Show exploratory plots
SAVE_PDF         = True              # Save all plots to PDF
PDF_PATH         = f"{TICKER}_risk_regimes.pdf"  # Output PDF file

# Number of HMM regimes and quantile thresholds
N_STATES         = 4                 # Number of hidden regimes
SEED             = 42                # RNG seed for reproducibility

# Quantile thresholds for defining extreme regimes (0 < q < 1)
EXTREME_BEAR_Q   = 0.1              # <=10th percentile => extreme_bear
EXTREME_BULL_Q   = 0.9              # >=90th percentile => extreme_bull

# Feature windows / toggles
VOL_Z_WINDOW     = 30                # Days for volume z-score
VOL_WINDOW       = 20                # Days for realized volatility (rolling stdev)
USE_VIX          = True              # Include ^VIX daily return
VIX_TICKER       = "^VIX"           # External volatility index ticker
USE_SPREAD       = True              # Include bid-ask spread proxy
SPREAD_METHOD    = "high_low"       # Only (High-Low)/Close currently
USE_SENTIMENT    = False             # Include sentiment score feature
SENTIMENT_PATH   = "sentiment.csv"  # CSV w/ Date, sentiment columns

# The exact order you want the regimes to appear
REGIME_ORDER = ['extreme_bull', 'bull', 'bear', 'extreme_bear']

# A matching color for each regime:
COLOR_MAP = {
    'extreme_bull':  '#008000',  
    'bull':          '#ACF3AE',  
    'bear':          '#FA6B84',  
    'extreme_bear':  '#990000',  
}

# -------------------------------------------------------------
# Dependencies: pip install yfinance hmmlearn pandas numpy matplotlib
# If USE_SENTIMENT == True: pip install vaderSentiment
# -------------------------------------------------------------
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from scipy.stats import skew, kurtosis
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

# 3b. State-to-Regime Mapping with Quantile Thresholds
# -------------------------------------------------------------
def _map_states_to_regimes(hmm: GaussianHMM) -> dict:
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
# 4. Classification pipeline (with posterior probabilities)
# -------------------------------------------------------------
def classify_risk_regimes() -> pd.DataFrame:
    raw = _download_yf(TICKER, START_DATE, END_DATE)
    feat_df = _feature_engineering(raw)
    X = feat_df.values

    # fit HMM and compute posteriors
    model = _fit_hmm(X, N_STATES, SEED)
    posteriors = model.predict_proba(X)  # shape (n_samples, n_states)

    # map states to regimes
    mapping = _map_states_to_regimes(model)
    state_seq = model.predict(X)
    regimes = np.vectorize(mapping.get)(state_seq)

    # assemble output DataFrame
    out = raw.loc[feat_df.index].copy()
    out[feat_df.columns] = feat_df
    out['regime'] = regimes
    out['regime_num'] = pd.Categorical(regimes, categories=sorted(set(mapping.values())), ordered=True).codes

    # add posterior probability columns per state label
    for state, label in mapping.items():
        out[f'prob_{label}'] = posteriors[:, state]

        # plotting & PDF
    # Save plots to PDF if requested
    if SAVE_PDF:
        with PdfPages(PDF_PATH) as pdf:
            _plot_summary_stats(out, pdf)
            _plot_transition_matrix(out, pdf)
            _plot_price_shade(out, pdf)
            _plot_posteriors(out, pdf)
            _plot_return_hist(out, pdf)
            _plot_scatter_feats(out, feat_df.columns, pdf)
    # Show plots interactively if requested
    if PLOT:
        _plot_summary_stats(out)
        _plot_transition_matrix(out)
        _plot_price_shade(out)
        _plot_posteriors(out)
        _plot_return_hist(out)
        _plot_scatter_feats(out, feat_df.columns)
    return out

# -------------------------------------------------------------
# 5. Plotting helpers (with PDF saving)
# -------------------------------------------------------------

def _plot_price_shade(df: pd.DataFrame, pdf: PdfPages = None):
    """Price chart with regime shading."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['Close'], label='Close')
    for regime in REGIME_ORDER:
        mask = (df['regime'] == regime)
        if not mask.any(): 
            continue
        ax.fill_between(df.index,
                        df['Close'].min(), df['Close'].max(),
                        where=mask,
                        color=COLOR_MAP[regime],
                        alpha=0.2,
                        label=regime)
    ax.set_title(f"{TICKER}: Price with Regime Shading")
    ax.legend()
    if pdf: pdf.savefig(fig)
    if PLOT: plt.show()
    plt.close(fig)

def _plot_return_hist(df: pd.DataFrame, pdf: PdfPages = None):
    # 1) create fig + ax
    fig, ax = plt.subplots(figsize=(12, 5))

    # 2) loop in fixed order and use same colors
    for regime in REGIME_ORDER:
        data = df.loc[df['regime'] == regime, 'log_ret']
        if data.empty:
            continue
        ax.hist(
            data,
            bins=50,
            alpha=0.6,
            label=regime,
            color=COLOR_MAP[regime],
        )

    # 3) decorate
    ax.set_title(f"{TICKER}: Log-Return Distribution by Regime")
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")

    # 4) save/show/close
    if pdf:
        pdf.savefig(fig)
    if PLOT:
        plt.show()
    plt.close(fig)

def _plot_scatter_feats(df: pd.DataFrame, feat_cols: list, pdf: PdfPages = None):
    """Pairwise scatter plots of features by regime."""
    for x, y in itertools.combinations(feat_cols, 2):
        fig, ax = plt.subplots(figsize=(6, 4))

        # plot *all* regimes on this same fig
        for regime in REGIME_ORDER:
            mask = (df['regime'] == regime)
            if not mask.any():
                continue
            ax.scatter(
                df.loc[mask, x],
                df.loc[mask, y],
                alpha=0.6,
                label=regime,
                color=COLOR_MAP[regime]
            )

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{x} vs {y}")
        ax.legend()
        plt.tight_layout()

        if pdf:
            pdf.savefig(fig)
        if PLOT:
            plt.show()
        plt.close(fig)

def _plot_posteriors(df: pd.DataFrame, pdf: PdfPages = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    dates = df.index

    # Build list of arrays in REGIME_ORDER
    prob_vals = [df[f'prob_{r}'] for r in REGIME_ORDER]
    colors   = [COLOR_MAP[r]        for r in REGIME_ORDER]

    ax.stackplot(dates,
                 *prob_vals,
                 labels=REGIME_ORDER,
                 colors=colors,
                 alpha=0.6)
    ax.set_title(f"{TICKER}: State Posterior Probabilities")
    ax.legend(loc='upper left')
    ax.set_ylabel('Probability')
    if pdf: pdf.savefig(fig)
    if PLOT: plt.show()
    plt.close(fig)

# -----------------------------------------------------------------------------
# SUMMARY-PAGE PLOTTER (FIRST PAGE IN PDF)
# -----------------------------------------------------------------------------

def _plot_summary_stats(df: pd.DataFrame, pdf: PdfPages | None = None):
    """First PDF page: key regime diagnostics (table only)."""
    # Compute regime-level statistics
    results = []
    for regime in REGIME_ORDER:
        mask = df['regime'] == regime
        if not mask.any():
            continue
        # choose return column
        ret_col = 'ret' if 'ret' in df.columns else 'log_ret'
        rets = df.loc[mask, ret_col]
        count = mask.sum()
        mean_ret = rets.mean()
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = mean_ret / (ann_vol + 1e-9)
        downside_std = rets[rets < 0].std() * np.sqrt(252)
        sortino = mean_ret / (downside_std + 1e-9)
        cum = (1 + rets).cumprod()
        drawdown = (cum - cum.cummax()) / cum.cummax()
        max_dd = drawdown.min()
        sk = skew(rets)
        kt = kurtosis(rets)
        z_95 = 1.64485
        var95 = -(mean_ret + z_95 * rets.std())
        es95 = -mean_ret + rets.std() * (np.exp(-z_95**2/2)/(np.sqrt(2*np.pi)*(1-0.95)))
        post_col = df.get(f'prob_{regime}', pd.Series(index=df.index, data=np.nan))
        avg_post = post_col.mean()
        last_post = post_col.iloc[-1] if not post_col.empty else np.nan
        transitions = df['regime'].shift().fillna(regime).astype(str) + '_' + df['regime'].astype(str)
        n_starts = (transitions == f'{regime}_{regime}').sum()
        exp_dur = count / max(n_starts, 1)
        results.append([
            regime,
            count,
            round(mean_ret, 4),
            round(ann_vol, 4),
            round(sharpe, 2),
            round(sortino, 2),
            round(max_dd, 2),
            round(sk, 2),
            round(kt, 2),
            round(var95, 4),
            round(es95, 4),
            round(avg_post, 3),
            round(last_post, 3),
            round(exp_dur, 1)
        ])
    col_labels = [
        'Regime', 'Count', 'Mean', 'Ann σ', 'Sharpe', 'Sortino', 'MaxDD',
        'Skew', 'Kurt', 'VaR95', 'ES95', 'Avg Post', 'Last Post', 'Exp Dur'
    ]

    # Build table figure
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    tbl = ax.table(
        cellText=results,
        colLabels=col_labels,
        loc='upper center',
        cellLoc='center',
        colWidths=[0.07] + [0.06]*13
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)

    plt.tight_layout()
    if pdf:
        pdf.savefig(fig)
    if PLOT:
        plt.show()
    plt.close(fig)

# -----------------------------------------------------------------------------
# TRANSITION MATRIX PAGE WITH FORECASTS & LOOK-BACKS (SECOND PAGE IN PDF)
# -----------------------------------------------------------------------------

def _plot_transition_matrix(df: pd.DataFrame, pdf: PdfPages | None = None):
    """Second PDF page: current state, forecasts, look-backs, and centered transition heatmap with axis labels."""
    regimes = REGIME_ORDER
    n = len(regimes)

    # Empirical transition matrix
    trans_counts = np.zeros((n, n))
    prev = df['regime'].iloc[0]
    for curr in df['regime'].iloc[1:]:
        i, j = regimes.index(prev), regimes.index(curr)
        trans_counts[i, j] += 1
        prev = curr
    trans_prob = trans_counts / trans_counts.sum(axis=1, keepdims=True)

    # Identify current state
    current_state = df['regime'].iloc[-1]

    # Last posterior probabilities
    last_probs = np.array([df[f'prob_{r}'].iloc[-1] for r in regimes])

    forecast_steps = [1, 5, 20, 50]
    lookback_days = [1, 5, 20, 50]
    txt_lines = []

    # Current state line
    txt_lines.append(f"Current state → {current_state}")
    txt_lines.append("")

    # Forward multi-step forecasts
    for k in forecast_steps:
        tp_k = np.linalg.matrix_power(trans_prob, k) if k > 1 else trans_prob
        probs_k = last_probs @ tp_k
        mode_k = regimes[int(np.argmax(probs_k))]
        txt_lines.append(
            f"{k}-step forecast → {mode_k} | " + 
            ", ".join([f"{r}: {p:.2f}" for r, p in zip(regimes, probs_k)])
        )
    txt_lines.append("")

    # 1-step forecasts from past days
    for d in lookback_days:
        if d <= len(df):
            posterior_d = np.array([df[f'prob_{r}'].iloc[-d] for r in regimes])
            f1 = posterior_d @ trans_prob
            mode1 = regimes[int(np.argmax(f1))]
            txt_lines.append(
                f"1-step forecast from {d}-day ago → {mode1} | " + 
                ", ".join([f"{r}: {p:.2f}" for r, p in zip(regimes, f1)])
            )
    txt_lines.append("")

    # 20-step forecasts from past days
    for d in lookback_days:
        if d <= len(df):
            posterior_d = np.array([df[f'prob_{r}'].iloc[-d] for r in regimes])
            f20 = posterior_d @ np.linalg.matrix_power(trans_prob, 20)
            mode20 = regimes[int(np.argmax(f20))]
            txt_lines.append(
                f"20-step forecast from {d}-day ago → {mode20} | " + 
                ", ".join([f"{r}: {p:.2f}" for r, p in zip(regimes, f20)])
            )
    
    # Join lines into text block
    text_block = "\n".join(txt_lines)

    # Plot text and heatmap
    fig, (ax_txt, ax_hm) = plt.subplots(
        2, 1,
        figsize=(11, 8.5),
        gridspec_kw={'height_ratios': [0.4, 0.6]}
    )
    # Text panel
    ax_txt.axis('off')
    ax_txt.text(
        0.5, 0.5,
        text_block,
        ha='center', va='center',
        fontsize=10,
        family='monospace'
    )

    # Heatmap panel (centered)
    im = ax_hm.imshow(trans_prob, cmap='Blues', vmin=0, vmax=1)
    ax_hm.set_anchor('C')
    ax_hm.set_aspect('equal')
    ax_hm.set_xticks(range(n), regimes, rotation=45, ha='center')
    ax_hm.set_yticks(range(n), regimes, va='center')
    ax_hm.set_xlabel('Ending State', labelpad=10, fontsize=12)
    ax_hm.set_ylabel('Starting State', labelpad=10, fontsize=12)
    ax_hm.set_title('Empirical Transition Probabilities', fontsize=14)
    for i in range(n):
        for j in range(n):
            ax_hm.text(j, i, f"{trans_prob[i, j]:.2f}", ha='center', va='center')
    fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if pdf: pdf.savefig(fig)
    if PLOT: plt.show()
    plt.close(fig)


# -------------------------------------------------------------
if __name__ == "__main__":
    df = classify_risk_regimes()
    # Safely gather probability columns (ensure string names)
    prob_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('prob_')]
    print(df.tail()[['Close', 'regime'] + prob_cols])
