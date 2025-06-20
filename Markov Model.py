# Markov Model.py – sticky HMM with adaptive detection
# ----------------------------------------------------------------------------- 
# Detects market regimes via a Hidden Markov Model (HMM) with:
#   • Sticky‑state persistence
#   • Adaptive rolling refits + probability triggers
# Robustness fixes ensure numerical stability (no NaNs in startprob_ / transmat_).
# -----------------------------------------------------------------------------

import os
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM, GMMHMM  # GMMHMM: Gaussian mixture HMM (for heavy tails)
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.linalg as linalg

# Avoid MKL+KMeans memory‑leak warning on Windows (harmless but noisy)
os.environ.setdefault("OMP_NUM_THREADS", "1")

# === USER CONFIGURATION =======================================================
TICKER              = "SPY"
START_DATE          = "2015-01-01"
END_DATE            = None            # None ⇒ today
PLOT                = False
SAVE_PDF            = True
REPORT_DIR          = "Reports"

# HMM / Regime ----------------------------------------------------------------
# MODEL_TYPE selects emission model:
#   • "GaussianHMM" → single Gaussian emission (default: simpler, faster)
#   • "GMMHMM"     → Gaussian mixture emissions (better for fat-tailed regimes)
MODEL_TYPE         = "GaussianHMM"   # choose between "GaussianHMM" or "GMMHMM"
N_MIX              = 2                # number of mixtures for GMMHMM (ignored by GaussianHMM)

N_STATES            = 5
SEED                = 42
STICKINESS          = 5.0             # Dirichlet prior boost on self-transitions
MIN_SELF_PROB       = 0.0             # 0 ⇒ off
COVARIANCE_TYPE     = "full"          # "diag" safe; switch to "full" if desired
MIN_COVAR           = 1e-3            # Regularisation for covariance matrices

# Adaptive detection ----------------------------------------------------------
ADAPTIVE_WINDOW     = 252             # rolling look‑back (≈1 year)
ADAPTIVE_FREQ       = 5               # refit every N days
PROB_THRESH         = 0.6             # posterior threshold
PROB_CONSEC_DAYS    = 2               # consecutive confirmations

# Regime mapping --------------------------------------------------------------
EXTREME_BEAR_Q      = 0.10
EXTREME_BULL_Q      = 0.90
NEUTRAL_PCT         = 0.35            # 0 ⇒ disable neutral regime

REGIME_ORDER        = [
    "extreme_bull", "bull", "neutral", "bear", "extreme_bear"
]
COLOR_MAP           = {
    "extreme_bull": "#008000", "bull": "#ACF3AE", "neutral": "#C0C0C0",
    "bear": "#FA6B84", "extreme_bear": "#990000",
}

# Feature toggles -------------------------------------------------------------
VOL_Z_WINDOW        = 30
VOL_WINDOW          = 20
USE_VIX             = True
VIX_TICKER          = "^VIX"
USE_SPREAD          = True
SPREAD_METHOD       = "high_low"
USE_SENTIMENT       = False
SENTIMENT_PATH      = "sentiment.csv"

# Output path -----------------------------------------------------------------
os.makedirs(REPORT_DIR, exist_ok=True)
PDF_PATH = os.path.join(
    REPORT_DIR,
    f"{TICKER}_risk_regimes_{datetime.now().strftime('%Y-%m-%d')}.pdf",
)

# ----------------------------------------------------------------------------- 
# Helpers
# -----------------------------------------------------------------------------
def _download_yf(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data for {ticker}.")
    return df

def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["log_ret"]  = np.log(df["Close"]).diff()
    vol_mu = df["Volume"].rolling(VOL_Z_WINDOW).mean()
    vol_sd = df["Volume"].rolling(VOL_Z_WINDOW).std()
    f["vol_z"]    = (df["Volume"] - vol_mu) / vol_sd
    f["real_vol"] = f["log_ret"].rolling(VOL_WINDOW).std() * np.sqrt(252)
    if USE_SPREAD and SPREAD_METHOD == "high_low":
        f["spread"] = (df["High"] - df["Low"]) / df["Close"]
    if USE_VIX:
        vix = _download_yf(VIX_TICKER, START_DATE, END_DATE)
        f["vix_ret"] = np.log(vix["Close"]).diff().reindex(df.index)
    if USE_SENTIMENT:
        sent = pd.read_csv(SENTIMENT_PATH, parse_dates=["Date"], index_col="Date")
        f = f.join(sent, how="left")
    return f.dropna()

# ----------------------------------------------------------------------------- 
# Model fitting & sanitisation ------------------------------------------------ 
# -----------------------------------------------------------------------------
def _sanitize_startprob(model):
    sp = model.startprob_
    if (not np.isfinite(sp).all()) or sp.sum() == 0:
        model.startprob_ = np.full_like(sp, 1.0 / len(sp))
    else:
        model.startprob_ = sp / sp.sum()

def _sanitize_transmat(model):
    tm = model.transmat_.copy()
    tm[~np.isfinite(tm)] = 0.0
    row_sums = tm.sum(axis=1, keepdims=True)
    zero = row_sums.squeeze() == 0
    tm[zero] = 1.0 / tm.shape[1]
    model.transmat_ = tm / tm.sum(axis=1, keepdims=True)

def _sanitize_gmm_weights(model):
    if not hasattr(model, "weights_"):
        return
    w = model.weights_.copy()
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    row_sums = w.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze() == 0
    w[zero_rows] = 1.0 / w.shape[1]
    model.weights_ = w / w.sum(axis=1, keepdims=True)

def _sanitize_means(model):
    if hasattr(model, "means_"):
        model.means_ = np.nan_to_num(model.means_, nan=0.0, posinf=0.0, neginf=0.0)

def _sanitize_covars(model):
    if not hasattr(model, "covars_"):
        return
    
    MIN_COVAR = globals().get("MIN_COVAR", 1e-6)
    
    if model.covariance_type == "diag":
        if isinstance(model, GMMHMM):
            expected_shape = (model.n_components, model.n_mix, model.n_features)
        else:
            expected_shape = (model.n_components, model.n_features)
        if model.covars_.shape != expected_shape:
            print(f"Warning: covars_ shape {model.covars_.shape} does not match "
                  f"expected {expected_shape}, reinitializing with MIN_COVAR")
            model.covars_ = np.full(expected_shape, MIN_COVAR)
        else:
            model.covars_ = np.nan_to_num(model.covars_, nan=MIN_COVAR, posinf=MIN_COVAR, neginf=MIN_COVAR)
            model.covars_ = np.maximum(model.covars_, MIN_COVAR)
            if np.any(model.covars_ <= 0):
                print("Warning: Some variances are non-positive, setting to MIN_COVAR")
                model.covars_[model.covars_ <= 0] = MIN_COVAR
    elif model.covariance_type == "full":
        for i in range(model.n_components):
            for j in range(model.n_mix if isinstance(model, GMMHMM) else 1):
                cov = model.covars_[i, j].copy() if isinstance(model, GMMHMM) else model.covars_[i].copy()
                if not np.all(np.isfinite(cov)):
                    cov = np.eye(model.n_features) * MIN_COVAR
                else:
                    try:
                        linalg.cholesky(cov)
                    except linalg.LinAlgError:
                        epsilon = MIN_COVAR
                        while True:
                            try:
                                linalg.cholesky(cov + epsilon * np.eye(model.n_features))
                                break
                            except linalg.LinAlgError:
                                epsilon *= 10
                        cov += epsilon * np.eye(model.n_features)
                if isinstance(model, GMMHMM):
                    model.covars_[i, j] = cov
                else:
                    model.covars_[i] = cov

def _fit_hmm(X: np.ndarray, k: int, seed: int):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.maximum(std, 1e-8)
    X_std = (X - mean) / std

    prior = np.full((k, k), 1.0)
    np.fill_diagonal(prior, STICKINESS)

    def _make_model(reg_covar):
        if MODEL_TYPE == "GaussianHMM":
            return GaussianHMM(
                n_components=k,
                covariance_type=COVARIANCE_TYPE,
                n_iter=1000,
                random_state=seed,
                transmat_prior=prior,
                startprob_prior=np.full(k, 1.0),
                min_covar=reg_covar,
            )
        elif MODEL_TYPE == "GMMHMM":
            return GMMHMM(
                n_components=k,
                n_mix=N_MIX,
                covariance_type=COVARIANCE_TYPE,
                n_iter=1000,
                random_state=seed,
                transmat_prior=prior,
                startprob_prior=np.full(k, 1.0),
                min_covar=reg_covar,
            )
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    reg = MIN_COVAR
    for attempt in range(4):
        model = _make_model(reg)
        try:
            model.fit(X_std)
            break
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            reg *= 10
            if attempt == 3:
                raise RuntimeError("HMM failed to converge after retries.") from e

    _sanitize_startprob(model)
    _sanitize_transmat(model)
    if MODEL_TYPE == "GMMHMM":
        _sanitize_gmm_weights(model)
    _sanitize_means(model)
    _sanitize_covars(model)

    if MIN_SELF_PROB > 0.0:
        tm = model.transmat_.copy()
        diag = np.diag(tm)
        low = diag < MIN_SELF_PROB
        if low.any():
            tm[low] = tm[low] * (1.0 - MIN_SELF_PROB) / np.clip(1.0 - diag[low], 1e-9, None)
            np.fill_diagonal(tm, np.maximum(diag, MIN_SELF_PROB))
            model.transmat_ = tm / tm.sum(axis=1, keepdims=True)
    return model, mean, std

# ----------------------------------------------------------------------------- 
# Regime mapping -------------------------------------------------------------- 
# -----------------------------------------------------------------------------
def _map_states_to_regimes(hmm):
    if hasattr(hmm, "weights_"):
        mix_means = hmm.means_[:, :, 0]
        weights = hmm.weights_
        mu = np.sum(mix_means * weights, axis=1)
    else:
        mu = hmm.means_[:, 0]
    q_lo, q_hi = np.quantile(mu, EXTREME_BEAR_Q), np.quantile(mu, EXTREME_BULL_Q)
    use_neu = NEUTRAL_PCT > 0 and N_STATES >= 5
    if use_neu:
        half = NEUTRAL_PCT / 2
        qn_lo = np.quantile(mu, 0.5 - half)
        qn_hi = np.quantile(mu, 0.5 + half)
    med = np.median(mu)
    mapping = {}
    for s, m in enumerate(mu):
        if m <= q_lo:
            mapping[s] = "extreme_bear"
        elif m >= q_hi:
            mapping[s] = "extreme_bull"
        else:
            if use_neu and qn_lo <= m <= qn_hi:
                mapping[s] = "neutral"
            else:
                mapping[s] = "bear" if m < med else "bull"
    return mapping

# ----------------------------------------------------------------------------- 
# Adaptive signal generation -------------------------------------------------- 
# -----------------------------------------------------------------------------
def generate_adaptive_signals(feat: pd.DataFrame) -> pd.DataFrame:
    signals = []
    consec = 0
    last_lab = None
    model = None
    mean = None
    std = None

    for i in range(ADAPTIVE_WINDOW, len(feat)):
        if model is None or (i - ADAPTIVE_WINDOW) % ADAPTIVE_FREQ == 0:
            X_train = feat.iloc[i - ADAPTIVE_WINDOW:i].values
            model, mean, std = _fit_hmm(X_train, N_STATES, SEED)
            mapping = _map_states_to_regimes(model)

        x_new = feat.iloc[i].values
        x_new_std = (x_new - mean) / std
        x_new_std = x_new_std.reshape(1, -1)
        for attempt in range(4):
            try:
                probs = model.predict_proba(x_new_std)[0]
                break
            except Exception as e:
                print(f"Attempt {attempt + 1}: predict_proba failed with error: {e}")
                globals()["MIN_COVAR"] *= 10
                model, mean, std = _fit_hmm(feat.iloc[i - ADAPTIVE_WINDOW:i].values, N_STATES, SEED)
                mapping = _map_states_to_regimes(model)
        else:
            raise RuntimeError("Failed to evaluate probability after retries.")

        top_i = int(np.argmax(probs))
        label = mapping[top_i]

        if probs[top_i] >= PROB_THRESH:
            consec = consec + 1 if label == last_lab else 1
        else:
            consec = 0

        if consec >= PROB_CONSEC_DAYS:
            signals.append({
                "date": feat.index[i],
                "regime": label,
                "prob": probs[top_i],
            })
        last_lab = label

    return pd.DataFrame(signals)

# ----------------------------------------------------------------------------- 
# Core pipeline --------------------------------------------------------------- 
# -----------------------------------------------------------------------------
def classify_risk_regimes() -> pd.DataFrame:
    raw = _download_yf(TICKER, START_DATE, END_DATE)
    feat = _feature_engineering(raw)

    # Adaptive detection signals
    signals_df = generate_adaptive_signals(feat)
    print("Adaptive detection signals:\n", signals_df)

    X = feat.values
    model, mean, std = _fit_hmm(X, N_STATES, SEED)
    X_std = (X - mean) / std
    post = model.predict_proba(X_std)

    mapping = _map_states_to_regimes(model)
    states = model.predict(X_std)
    regimes = np.vectorize(mapping.get)(states)

    out = raw.loc[feat.index].copy()
    out[feat.columns] = feat
    out["regime"] = regimes
    out["regime_num"] = pd.Categorical(
        regimes,
        categories=list(dict.fromkeys(REGIME_ORDER)),
        ordered=True
    ).codes

    for s, label in mapping.items():
        out[f"prob_{label}"] = post[:, s]
    for r in REGIME_ORDER:
        col = f"prob_{r}"
        if col not in out.columns:
            out[col] = 0.0

    if SAVE_PDF:
        with PdfPages(PDF_PATH) as pdf:
            _plot_summary_stats(out, pdf)
            _plot_transition_matrix(out, pdf)
            _plot_price_shade(out, pdf)
            _plot_posteriors(out, pdf)
            _plot_return_hist(out, pdf)
            _plot_scatter_feats(out, feat.columns, pdf)

    return out

# ----------------------------------------------------------------------------- 
# Plotting helpers (omitted for brevity) --------------------------------------- 
# -----------------------------------------------------------------------------
def _plot_price_shade(df: pd.DataFrame, pdf: PdfPages | None = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df["Close"], label="Close")
    for regime in REGIME_ORDER:
        mask = df["regime"] == regime
        if not mask.any():
            continue
        ax.fill_between(
            df.index,
            df["Close"].min(),
            df["Close"].max(),
            where=mask,
            color=COLOR_MAP[regime],
            alpha=0.2,
            label=regime,
        )
    ax.set_title(f"{TICKER}: Price with Regime Shading")
    ax.legend()
    if pdf:
        pdf.savefig(fig)
    if PLOT:
        plt.show()
    plt.close(fig)

def _plot_return_hist(df: pd.DataFrame, pdf: PdfPages | None = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    for regime in REGIME_ORDER:
        data = df.loc[df["regime"] == regime, "log_ret"]
        if data.empty:
            continue
        ax.hist(data, bins=50, alpha=0.6, label=regime, color=COLOR_MAP[regime])
    ax.set_title(f"{TICKER}: Log‑Return Distribution by Regime")
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")
    if pdf:
        pdf.savefig(fig)
    if PLOT:
        plt.show()
    plt.close(fig)

def _plot_scatter_feats(df: pd.DataFrame, feat_cols: list[str], pdf: PdfPages | None = None):
    for x, y in itertools.combinations(feat_cols, 2):
        fig, ax = plt.subplots(figsize=(6, 4))
        for regime in REGIME_ORDER:
            mask = df["regime"] == regime
            if not mask.any():
                continue
            ax.scatter(
                df.loc[mask, x],
                df.loc[mask, y],
                alpha=0.6,
                label=regime,
                color=COLOR_MAP[regime],
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

def _plot_posteriors(df: pd.DataFrame, pdf: PdfPages | None = None):
    regimes_present = [r for r in REGIME_ORDER if f"prob_{r}" in df]
    prob_vals = [df[f"prob_{r}"] for r in regimes_present]
    colors = [COLOR_MAP[r] for r in regimes_present]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(df.index, *prob_vals, labels=regimes_present, colors=colors, alpha=0.6)
    ax.set_title(f"{TICKER}: State Posterior Probabilities")
    ax.set_ylabel("Probability")
    ax.legend(loc="upper left")
    if pdf:
        pdf.savefig(fig)
    if PLOT:
        plt.show()
    plt.close(fig)

# ----------------------------------------------------------------------------- 
# SUMMARY‑PAGE PLOTTER (FIRST PAGE IN PDF) 
# -----------------------------------------------------------------------------
def _plot_summary_stats(df: pd.DataFrame, pdf: PdfPages | None = None):
    results = []
    for regime in REGIME_ORDER:
        mask = df["regime"] == regime
        if not mask.any():
            continue
        ret_col = "ret" if "ret" in df.columns else "log_ret"
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
        es95 = -mean_ret + rets.std() * (np.exp(-z_95**2 / 2) / (np.sqrt(2 * np.pi) * (1 - 0.95)))
        post_col = df[f"prob_{regime}"] if f"prob_{regime}" in df else pd.Series(index=df.index, data=np.nan)
        avg_post = post_col.mean()
        last_post = post_col.iloc[-1] if not post_col.empty else np.nan
        transitions = df["regime"].shift().fillna(regime).astype(str) + "_" + df["regime"].astype(str)
        n_starts = (transitions == f"{regime}_{regime}").sum()
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
            round(exp_dur, 1),
        ])

    col_labels = [
        "Regime",
        "Count",
        "Mean",
        "Ann σ",
        "Sharpe",
        "Sortino",
        "MaxDD",
        "Skew",
        "Kurt",
        "VaR95",
        "ES95",
        "Avg Post",
        "Last Post",
        "Exp Dur",
    ]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    # Add a big page title
    date_end = END_DATE if END_DATE else pd.Timestamp.today().strftime("%Y-%m-%d")
    fig.suptitle(
        f"{TICKER} Risk Regime Analysis  ({START_DATE} → {date_end})",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )
    ax.axis("off")
    
    tbl = ax.table(
        cellText=results,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
        colWidths=[0.07] + [0.06] * 13,
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
# TRANSITION MATRIX PAGE (SECOND PAGE IN PDF) 
# -----------------------------------------------------------------------------
def _plot_transition_matrix(df: pd.DataFrame, pdf: PdfPages | None = None):
    regimes = [r for r in REGIME_ORDER if r in df["regime"].unique()]
    n = len(regimes)

    # empirical transition counts & probabilities
    trans_counts = np.zeros((n, n))
    prev = df["regime"].iloc[0]
    for curr in df["regime"].iloc[1:]:
        if prev not in regimes or curr not in regimes:
            prev = curr
            continue
        i, j = regimes.index(prev), regimes.index(curr)
        trans_counts[i, j] += 1
        prev = curr
    with np.errstate(invalid="ignore", divide="ignore"):
        trans_prob = np.nan_to_num(trans_counts / trans_counts.sum(axis=1, keepdims=True))

    # current state & last posteriors
    current_state = df["regime"].iloc[-1]
    last_probs = np.array([df[f"prob_{r}"].iloc[-1] for r in regimes])

    forecast_steps = [1, 5, 20, 50]
    lookback_days  = [1, 5, 20, 50]
    txt_lines = []

    # current state
    txt_lines.append(f"Current state → {current_state}")
    txt_lines.append("")

    # forward forecasts
    for k in forecast_steps:
        tp_k = np.linalg.matrix_power(trans_prob, k) if k > 1 else trans_prob
        probs_k = last_probs @ tp_k
        mode_k = regimes[int(np.argmax(probs_k))]
        txt_lines.append(
            f"{k}-step forecast → {mode_k} | " + 
            ", ".join([f"{r}: {p:.2f}" for r, p in zip(regimes, probs_k)])
        )
    txt_lines.append("")

    # backward‑looking one‑step forecasts
    for d in lookback_days:
        if d <= len(df):
            posterior_d = np.array([df[f"prob_{r}"].iloc[-d] for r in regimes])
            f1 = posterior_d @ trans_prob
            mode1 = regimes[int(np.argmax(f1))]
            txt_lines.append(
                f"1‑step forecast from {d}-day ago → {mode1} | " + 
                ", ".join([f"{r}: {p:.2f}" for r, p in zip(regimes, f1)])
            )
    txt_lines.append("")

    # 20‑step forecasts from past days
    for d in lookback_days:
        if d <= len(df):
            posterior_d = np.array([df[f"prob_{r}"].iloc[-d] for r in regimes])
            f20 = posterior_d @ np.linalg.matrix_power(trans_prob, 20)
            mode20 = regimes[int(np.argmax(f20))]
            txt_lines.append(
                f"20‑step forecast from {d}-day ago → {mode20} | " + 
                ", ".join([f"{r}: {p:.2f}" for r, p in zip(regimes, f20)])
            )

    text_block = "\n".join(txt_lines)

    fig, (ax_txt, ax_hm) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [0.4, 0.6]})

    # text panel
    ax_txt.axis("off")
    ax_txt.text(0.5, 0.5, text_block, ha="center", va="center", fontsize=10, family="monospace")

    # heatmap
    im = ax_hm.imshow(trans_prob, cmap="Blues", vmin=0, vmax=1)
    ax_hm.set_anchor("C")
    ax_hm.set_aspect("equal")
    ax_hm.set_xticks(range(n), regimes, rotation=45, ha="center")
    ax_hm.set_yticks(range(n), regimes, va="center")
    ax_hm.set_xlabel("Ending State", labelpad=10, fontsize=12)
    ax_hm.set_ylabel("Starting State", labelpad=10, fontsize=12)
    ax_hm.set_title("Empirical Transition Probabilities", fontsize=14)
    for i in range(n):
        for j in range(n):
            ax_hm.text(j, i, f"{trans_prob[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if pdf:
        pdf.savefig(fig)
    if PLOT:
        plt.show()
    plt.close(fig)

# ----------------------------------------------------------------------------- 
# Main ------------------------------------------------------------------------ 
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    df = classify_risk_regimes()
    prob_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("prob_")]
    print(df.tail()[["Close", "regime"] + prob_cols])