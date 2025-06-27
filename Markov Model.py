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
TICKER              = "IWM"
START_DATE          = "2000-01-01"
END_DATE            = None            # None ⇒ today
PLOT                = False
SAVE_PDF            = True
REPORT_DIR          = "Reports"

# HMM / Regime ----------------------------------------------------------------
# MODEL_TYPE selects emission model:
#   • "GaussianHMM" → single Gaussian emission (default: simpler, faster)
#   • "GMMHMM"     → Gaussian mixture emissions (better for fat-tailed regimes)
MODEL_TYPE         = "GMMHMM"       # choose between "GaussianHMM" or "GMMHMM"
N_MIX              = 3                #2 = normal + tails, 3 more precise (some overfitting), 4 starts overfitting. # number of mixtures for GMMHMM (ignored by GaussianHMM) 

N_STATES            = 5
BINARY_MODE = (N_STATES == 2)
SEED                = 42
STICKINESS          = 100             # Dirichlet prior boost on self-transitions
MIN_SELF_PROB       = 0.0             # 0 ⇒ off
COVARIANCE_TYPE     = "full"          # "diag" safe; tied, or switch to "full" if desired
MIN_COVAR           = 1e-6            # Regularisation for covariance matrices

# Adaptive detection ----------------------------------------------------------
ADAPTIVE_WINDOW     = 60             # rolling look‑back (≈1 year)
ADAPTIVE_FREQ       = 5               # refit every N days
PROB_THRESH         = 0.90             # posterior threshold
PROB_CONSEC_DAYS    = 5               # consecutive confirmations

# Regime mapping --------------------------------------------------------------
EXTREME_BEAR_Q      = 0.10
EXTREME_BULL_Q      = 0.90
NEUTRAL_PCT         = 0.2            # 0 ⇒ disable neutral regime

REGIME_ORDER        = [
    "extreme_bull", "bull", "neutral", "bear", "extreme_bear"
]
COLOR_MAP           = {
    "extreme_bull": "#008000", "bull": "#ACF3AE", "neutral": "#C0C0C0",
    "bear": "#FA6B84", "extreme_bear": "#990000",
}
if BINARY_MODE:
    REGIME_ORDER = ["bull", "bear"]
    COLOR_MAP = {"bull": "#008000", "bear": "#d14343"}

# Feature toggles -------------------------------------------------------------
VOL_Z_WINDOW        = 30
VOL_WINDOW          = 20
USE_VIX             = True
VIX_TICKER          = "^VIX"
USE_TLT             = False
TLT_TICKER          = "^TNX"        # 20+ year Treasury ETF as rate proxy
USE_SKEW            = True
SKEW_TICKER         = "^SKEW"       # CBOE SKEW Index
USE_MOVE            = False         
MOVE_TICKER         = "^MOVE"       # ICE BofA MOVE Index, Tends to give False signals for bond related stuff.
USE_SPREAD          = True
SPREAD_METHOD       = "high_low"
USE_MOMENTUM        = True
MOM_PERIOD          = 20            # look‐back window (days) for momentum
USE_SENTIMENT       = False
SENTIMENT_PATH      = "sentiment.csv"

# Output path -----------------------------------------------------------------
os.makedirs(REPORT_DIR, exist_ok=True)
PDF_PATH = os.path.join(
    REPORT_DIR,
    f"{TICKER}_{START_DATE}_to_{END_DATE}risk_regimes_{datetime.now().strftime('%Y-%m-%d')}.pdf",
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
    if USE_MOMENTUM:
        f["mom"] = (
            df["Close"]         # today's price
            .pct_change(periods=MOM_PERIOD)  # (P_t / P_{t-N}) – 1
            .reindex(df.index)
        )
    if USE_TLT:
        tlt = _download_yf(TLT_TICKER, START_DATE, END_DATE)
    if USE_SKEW:
        skew_df = _download_yf(SKEW_TICKER, START_DATE, END_DATE)
        f["skew_ret"] = np.log(skew_df["Close"]).diff().reindex(df.index)
    if USE_MOVE:
        move_df = _download_yf(MOVE_TICKER, START_DATE, END_DATE)
        f["move_ret"] = np.log(move_df["Close"]).diff().reindex(df.index)
    if USE_SENTIMENT:
        sent = pd.read_csv(SENTIMENT_PATH, parse_dates=["Date"], index_col="Date")
        f = f.join(sent, how="left")
    return f.dropna()

def _active_regimes() -> list[str]:
    """Return the regimes the rest of the code should use."""
    if BINARY_MODE:                             # two-state model
        return ["bull", "bear"]
    base = ["extreme_bull", "bull", "bear", "extreme_bear"]
    if NEUTRAL_PCT > 0:
        base.insert(2, "neutral")
    return base

# ----------------------------------------------------------------------------- 
# Model fitting & sanitisation ------------------------------------------------ 
# -----------------------------------------------------------------------------
def force_spd(matrix: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """
    Make a covariance matrix symmetric-positive-definite by
    bumping the spectrum until the smallest eigenvalue ≥ `floor`.
    Works in-place and returns the fixed matrix.
    """
    # symmetrise first (numerical noise)
    matrix[:] = 0.5 * (matrix + matrix.T)

    # eigen-decomposition is cheaper than repeated Cholesky
    w, V = linalg.eigh(matrix, lower=True)
    if np.min(w) < floor:
        w = np.clip(w, floor, None)
        matrix[:] = (V * w) @ V.T

    # final cheap guarantee (add jitter if Cholesky still fails)
    jitter = floor
    while True:
        try:
            _ = linalg.cholesky(matrix, lower=True)
            break
        except linalg.LinAlgError:
            matrix += np.eye(matrix.shape[0]) * jitter
            jitter *= 10
    return matrix

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

def _sanitize_covars(model, floor: float = 1e-6):
    """Ensure every covariance in the model is SPD."""
    if getattr(model, "covariance_type", "diag") == "diag":
        # diag ⇒ just clamp variances
        model.covars_ = np.maximum(model.covars_, floor)
        return

    # full or tied
    if model.covariance_type == "tied":      # one per state
        for s in range(model.covars_.shape[0]):
            force_spd(model.covars_[s], floor)
    else:                                    # full GMM ⇒ state × mix
        for s in range(model.covars_.shape[0]):
            for m in range(model.covars_.shape[1]):
                force_spd(model.covars_[s, m], floor)


def _fit_hmm(X: np.ndarray, k: int, seed: int):
    import numpy as np
    from hmmlearn.hmm import GaussianHMM, GMMHMM
    from sklearn.mixture import GaussianMixture
    import scipy.linalg as linalg

    # 1. Standardize
    mean = np.mean(X, axis=0)
    std  = np.std(X, axis=0)
    std  = np.maximum(std, 1e-8)
    X_std = (X - mean) / std

    # 2. Build transition prior
    prior = np.full((k, k), 1.0)
    np.fill_diagonal(prior, STICKINESS)

    # 3. Determine model & covariance trial order
    model_order = [MODEL_TYPE]
    if MODEL_TYPE == "GMMHMM":
        model_order.append("GaussianHMM")
    else:
        model_order.append("GMMHMM")

    cov_order = [COVARIANCE_TYPE]
    alt_cov = "diag" if COVARIANCE_TYPE != "diag" else "full"
    cov_order.append(alt_cov)

    last_err = None

    # 4. Try every (model_type, cov_type) combo
    for mtype in model_order:
        for cov_type in cov_order:

            # 4a. If GMMHMM, pre-fit a GaussianMixture for emissions
            if mtype == "GMMHMM":
                total_comps = k * N_MIX
                gmm = GaussianMixture(
                    n_components=total_comps,
                    covariance_type=cov_type,
                    reg_covar=MIN_COVAR,
                    max_iter=200,
                    random_state=seed
                ).fit(X_std)

                n_feat = X_std.shape[1]
                means_hmm   = gmm.means_.reshape(k, N_MIX, n_feat)
                weights_hmm = gmm.weights_.reshape(k, N_MIX)
                # normalize
                weights_hmm /= weights_hmm.sum(axis=1, keepdims=True)

                if cov_type == "diag":
                    covars_hmm = gmm.covariances_.reshape(k, N_MIX, n_feat)
                else:
                    covars_hmm = gmm.covariances_.reshape(k, N_MIX, n_feat, n_feat)

            # 4b. Build factory for this combo
            def make_model(reg_covar):
                if mtype == "GaussianHMM":
                    return GaussianHMM(
                        n_components=k,
                        covariance_type=cov_type,
                        n_iter=1000,
                        random_state=seed,
                        transmat_prior=prior,
                        startprob_prior=np.full(k, 1.0),
                        min_covar=reg_covar,
                    )
                else:
                    return GMMHMM(
                        n_components=k,
                        n_mix=N_MIX,
                        covariance_type=cov_type,
                        n_iter=1000,
                        random_state=seed,
                        transmat_prior=prior,
                        startprob_prior=np.full(k, 1.0),
                        min_covar=reg_covar,
                    )

            # 4c. Try EM with increasing regularization
            reg = MIN_COVAR
            for attempt in range(4):
                model = make_model(reg)
                if mtype == "GMMHMM":
                    model.means_   = means_hmm
                    model.covars_  = covars_hmm
                    model.weights_ = weights_hmm
                    model.init_params = ""  # freeze emissions

                try:
                    model.fit(X_std)
                    _sanitize_covars(model)
                    # record successful combo
                    final_model_type = mtype
                    final_cov_type   = cov_type
                    break
                except Exception as e:
                    last_err = e
                    reg *= 10
            else:
                # this cov_type failed for this model_type
                continue

            # broke out of inner loop successfully
            break
        else:
            # this model_type failed all covariances
            continue
        # model_type succeeded
        break
    else:
        # all four combos failed
        raise RuntimeError(
            f"HMM failed for all combos {model_order}×{cov_order}, last error:\n{last_err}"
        )

    # 5. Final sanitization
    _sanitize_startprob(model)
    _sanitize_transmat(model)
    if final_model_type == "GMMHMM":
        _sanitize_gmm_weights(model)
    _sanitize_means(model)
    _sanitize_covars(model)

    # 6. Enforce minimum self-transition
    if MIN_SELF_PROB > 0.0:
        tm = model.transmat_.copy()
        diag_probs = np.diag(tm)
        low = diag_probs < MIN_SELF_PROB
        if low.any():
            tm[low] = tm[low] * (1.0 - MIN_SELF_PROB) / np.clip(1 - diag_probs[low], 1e-9, None)
            np.fill_diagonal(tm, np.maximum(diag_probs, MIN_SELF_PROB))
            model.transmat_ = tm / tm.sum(axis=1, keepdims=True)

    return model, mean, std

# ----------------------------------------------------------------------------- 
# Regime mapping -------------------------------------------------------------- 
# -----------------------------------------------------------------------------
def _map_states_to_regimes(hmm):
    if BINARY_MODE:
        # mean return for each hidden state
        if hasattr(hmm, "weights_"):                             # GMMHMM
            mu = np.sum(hmm.means_[:, :, 0] * hmm.weights_, axis=1)
        else:                                                    # GaussianHMM
            mu = hmm.means_[:, 0]

        lo, hi = np.argsort(mu)          # lower mean = bear, higher = bull
        return {lo: "bear", hi: "bull"}
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
                _sanitize_covars(model) 
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

    for r in _active_regimes():
        out[f"prob_{r}"] = 0.0
    for s, label in mapping.items():
        out[f"prob_{label}"] += post[:, s]
    prob_mat = out[[f"prob_{r}" for r in _active_regimes()]].values
    row_sums  = prob_mat.sum(axis=1, keepdims=True)
    out.loc[:, [f"prob_{r}" for r in _active_regimes()]] = np.divide(
        prob_mat, np.where(row_sums == 0, 1, row_sums)
    )


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
    for regime in _active_regimes():
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
    for regime in _active_regimes():
        data = df.loc[df["regime"] == regime, "log_ret"]
        if data.empty:
            continue
        ax.hist(data, bins=50, alpha=0.35, label=regime, color=COLOR_MAP[regime])
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
        for regime in _active_regimes():
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
    regimes_present = [r for r in _active_regimes() if df[f"prob_{r}"].max() > 1e-6]
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
# SUMMARY-PAGE PLOTTER (FIRST PAGE IN PDF) 
# -----------------------------------------------------------------------------
# ----------------------------------------------------------------------------- 
# UPDATED _plot_summary_stats WITH COMBINED STATE SUMMARY, AVG RUN RETURN & HIT RATE
# -----------------------------------------------------------------------------
def _plot_summary_stats(df: pd.DataFrame, pdf: PdfPages | None = None):
    # determine end date
    date_end = END_DATE if END_DATE else pd.Timestamp.today().strftime("%Y-%m-%d")
    # key parameters lines
    line1 = f"HMM: {MODEL_TYPE}, Cov: {COVARIANCE_TYPE}, N Mix: {N_MIX}, Stickiness: {STICKINESS}"
    line2 = f"Adaptive: window={ADAPTIVE_WINDOW}d, freq={ADAPTIVE_FREQ}d, thresh={PROB_THRESH:.2f}, days={PROB_CONSEC_DAYS:.2f}"
    line3 = f"Regime mapping: Q_lo={EXTREME_BEAR_Q}, Q_hi={EXTREME_BULL_Q}, NeuPct={NEUTRAL_PCT}"
    line4 = f"Features: Momentum={USE_MOMENTUM}, HighLow={USE_SPREAD}, VIX={USE_VIX}, SKEW={USE_SKEW}, TLT={USE_TLT}, MOVE={USE_MOVE}"

    # compute performance statistics (existing)
    results = []
    for regime in _active_regimes():
        mask = df["regime"] == regime
        if not mask.any(): continue
        ret_col = "ret" if "ret" in df.columns else "log_ret"
        rets = df.loc[mask, ret_col]
        count = mask.sum()
        mean_ret = rets.mean(); ann_ret = mean_ret * np.sqrt(252)
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = mean_ret / (ann_vol + 1e-9)
        neg = rets[rets < 0]
        downside = neg.std() * np.sqrt(252) if not neg.empty else 0.0
        sortino = mean_ret / (downside + 1e-9)
        cum = (1 + rets).cumprod(); max_dd = ((cum - cum.cummax())/cum.cummax()).min()
        sk = skew(rets); kt = kurtosis(rets)
        z95 = 1.64485
        var95 = -(mean_ret + z95 * rets.std())
        es95 = -mean_ret + rets.std() * (np.exp(-z95**2/2)/(np.sqrt(2*np.pi)*(1-0.95)))
        post = df.get(f"prob_{regime}", pd.Series(np.nan, index=df.index))
        avg_post = post.mean(); last_post = post.iloc[-1] if not post.empty else np.nan
        trans = df["regime"].shift().fillna(regime) + "_" + df["regime"]
        starts = (trans == f"{regime}_{regime}").sum()
        exp_dur = count / max(starts, 1)
        results.append([
            regime, count, round(ann_ret,4), round(ann_vol,4),
            round(sharpe,2), round(sortino,2), round(max_dd,2),
            round(sk,2), round(kt,2), round(var95,4),
            round(es95,4), round(avg_post,3), round(last_post,3), round(exp_dur,1)
        ])
    col_labels = [
        "Regime","Count","Ann Ret","Ann σ","Sharpe","Sortino",
        "MaxDD","Skew","Kurt","VaR95","ES95",
        "AvgPost","LastPost","ExpDur"
    ]

    # figure setup
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"{TICKER} Risk Regime Analysis  ({START_DATE} → {date_end})",fontsize=18, fontweight="bold", y=0.97)
    fig.text(0.02, 0.90, "Parameters:", ha="left", va="bottom", fontsize=11, fontweight="bold")
    fig.text(0.02, 0.87, "\n".join([line1, line2, line3, line4]),ha="left", va="top", fontsize=10, family="monospace")

    # first statistics table
    fig.text(0.02, 0.75, "Statistics", ha="left", va="bottom", fontsize=11, fontweight="bold")
    ax_tbl = fig.add_axes([0.02, 0.72, 0.96, 0.25])
    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=results,
        colLabels=col_labels,
        cellLoc="center",
        colWidths=[0.07] + [0.065]*13
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)

    # -------------------------------------------------------------------------
     # compute runs for state summaries
    runs = []  # list of (regime, start, end, duration_days, return)
    regime_series = df['regime']
    dates = df.index
    prev = regime_series.iloc[0]
    start = dates[0]
    for dt, curr in zip(dates[1:], regime_series.iloc[1:]):
        if curr != prev:
            end = dt
            duration = (end - start).days
            # extract scalar prices
            price_in = df['Close'].loc[start]
            price_in = float(price_in.iloc[0]) if hasattr(price_in, 'iloc') else float(price_in)
            price_out = df['Close'].loc[end]
            price_out = float(price_out.iloc[0]) if hasattr(price_out, 'iloc') else float(price_out)
            run_return = price_out/price_in - 1
            runs.append((prev, start, end, duration, run_return))
            prev = curr
            start = dt
    # current run
    duration = (dates[-1] - start).days
    price_in = df['Close'].loc[start]
    price_in = float(price_in.iloc[0]) if hasattr(price_in, 'iloc') else float(price_in)
    price_out = df['Close'].iloc[-1]
    price_out = float(price_out.iloc[0]) if hasattr(price_out, 'iloc') else float(price_out)
    run_return = price_out/price_in - 1
    runs.append((prev, start, None, duration, run_return))

    # build combined summary rows
    combined = []
    for regime in _active_regimes():
        state_runs = [r for r in runs if r[0] == regime]
        # length stats
        dur_list = [r[3] for r in state_runs]
        avg_len = int(np.mean(dur_list)) if dur_list else 0
        med_len = int(np.median(dur_list)) if dur_list else 0
        max_len = max(dur_list) if dur_list else 0
        min_len = min(dur_list) if dur_list else 0
        # run returns
        ret_list = [r[4] for r in state_runs]
        avg_run_ret = round(np.mean(ret_list),4) if ret_list else 0.0
        # hit rate
        if regime in ['bull','extreme_bull']:
            hits = sum(r > 0 for r in ret_list)
        else:
            hits = sum(r < 0 for r in ret_list)
        hit_rate = round(hits / len(ret_list),3) if ret_list else 0.0
        # last switch
        last_run = state_runs[-1] if state_runs else (regime, None, None, 0, 0)
        date_in = last_run[1].strftime('%Y-%m-%d') if last_run[1] else ''
        date_out = last_run[2].strftime('%Y-%m-%d') if last_run[2] else ''
        dur_days = last_run[3]
        next_state = ''
        if last_run[2] is not None:
            idx = df.index.get_loc(last_run[2])
            if idx+1 < len(df): next_state = df['regime'].iloc[idx+1]
        combined.append([
            regime, date_in, date_out, dur_days,
            round(last_run[4],4), next_state,
            avg_len, med_len, max_len, min_len,
            avg_run_ret, hit_rate
        ])
    combined_labels = [
        "Regime","DateIn","DateOut","DurDays","LastRet","NextState",
        "AvgLen","MedLen","MaxLen","MinLen","AvgRunRet","HitRate"
    ]

    # combined table
    fig.text(0.02, 0.50, "State Summary (Last Switch, Length & Performance)", ha="left", va="bottom",
             fontsize=11, fontweight="bold")
    ax_comb = fig.add_axes([0.02, 0.47, 0.96, 0.28])
    ax_comb.axis('off')
    tbl_comb = ax_comb.table(
        cellText=combined,
        colLabels=combined_labels,
        cellLoc='center',
        colWidths=[0.09,0.09,0.09,0.06,0.06,0.08,0.06,0.06,0.06,0.06,0.08,0.06]
    )
    tbl_comb.auto_set_font_size(False); tbl_comb.set_fontsize(8); tbl_comb.scale(1,1.2)

    plt.tight_layout()
    if pdf: pdf.savefig(fig)
    if PLOT: plt.show()
    plt.close(fig)

# ----------------------------------------------------------------------------- 
# TRANSITION MATRIX PAGE (SECOND PAGE IN PDF) 
# -----------------------------------------------------------------------------
def _plot_transition_matrix(df: pd.DataFrame, pdf: PdfPages | None = None):
    regimes = [r for r in _active_regimes() if r in df["regime"].unique()]
    n = len(regimes)

    # 1-step transition probabilities
    counts = np.zeros((n, n))
    prev = df["regime"].iloc[0]
    for curr in df["regime"].iloc[1:]:
        if prev in regimes and curr in regimes:
            i, j = regimes.index(prev), regimes.index(curr)
            counts[i, j] += 1
        prev = curr
    with np.errstate(divide="ignore", invalid="ignore"):
        trans_prob = np.nan_to_num(counts / counts.sum(axis=1, keepdims=True))

    # 20-step
    trans_prob_20 = np.linalg.matrix_power(trans_prob, 20)

    # text forecast block
    current = df["regime"].iloc[-1]
    last_p = np.array([df[f"prob_{r}"].iloc[-1] for r in regimes])
    txt = [f"Current state → {current}", ""]
    for k in (1,5,20,50):
        mat = trans_prob if k==1 else np.linalg.matrix_power(trans_prob, k)
        p = last_p @ mat
        mode = regimes[int(np.argmax(p))]
        txt.append(f"{k}-step forecast → {mode} | " +
                   ", ".join(f"{r}: {v:.2f}" for r,v in zip(regimes,p)))
    txt.append("")
    for d in (1,5,20,50):
        if d <= len(df):
            post_d = np.array([df[f"prob_{r}"].iloc[-d] for r in regimes])
            p1 = post_d @ trans_prob
            m1 = regimes[int(np.argmax(p1))]
            txt.append(f"1-step from {d}-day ago → {m1} | " +
                       ", ".join(f"{r}: {v:.2f}" for r,v in zip(regimes,p1)))
    txt.append("")
    for d in (1,5,20,50):
        if d <= len(df):
            post_d = np.array([df[f"prob_{r}"].iloc[-d] for r in regimes])
            p20 = post_d @ trans_prob_20
            m20 = regimes[int(np.argmax(p20))]
            txt.append(f"20-step from {d}-day ago → {m20} | " +
                       ", ".join(f"{r}: {v:.2f}" for r,v in zip(regimes,p20)))

    # layout: text + two heatmaps
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.4,0.6], width_ratios=[1,1])
    ax_txt = fig.add_subplot(gs[0, :])
    ax1   = fig.add_subplot(gs[1, 0])
    ax2   = fig.add_subplot(gs[1, 1])

    # text
    ax_txt.axis("off")
    ax_txt.text(0.5, 0.5, "\n".join(txt), ha="center", va="center",
                fontsize=10, family="monospace")

    # heatmaps (shared vmin/vmax)
    for ax, mat, title in [(ax1, trans_prob, "1-Step Transition Probabilities"),
                            (ax2, trans_prob_20, "20-Step Transition Probabilities")]:
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(n), regimes)
        ax.set_yticks(range(n), regimes)
        ax.tick_params(axis='x', labelrotation=45, labelsize=8, pad=10)
        ax.tick_params(axis='y', labelsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j,i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    if pdf: pdf.savefig(fig)
    if PLOT: plt.show()
    plt.close(fig)

# ----------------------------------------------------------------------------- 
# Main ------------------------------------------------------------------------ 
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    df = classify_risk_regimes()
    prob_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("prob_")]
    print(df.tail()[["Close", "regime"] + prob_cols])