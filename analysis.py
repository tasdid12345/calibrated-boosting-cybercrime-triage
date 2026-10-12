"""Baseline (default RandomForest) vs improved (class-balanced, CV-tuned, isotonic-calibrated LightGBM)
on the UCI Phishing Websites dataset, used as a stand-in for flagging malicious entities."""
import json, os, socket, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
import lightgbm as lgb
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

SEED = 42
SEEDS = [42, 43, 44]          # both models are run with these seeds
MAX_ROWS = 200_000            # subsample cap to keep runtime small
DATASET_ID = 327
SOURCE_URL = "https://archive.ics.uci.edu/dataset/327/phishing+websites"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data():
    """Fetch via ucimlrepo with a 60 s socket timeout and retry/backoff on failures (e.g. HTTP 429/5xx)."""
    socket.setdefaulttimeout(60)
    ds = None
    for attempt in range(5):
        try:
            ds = fetch_ucirepo(id=DATASET_ID)
            break
        except Exception as e:  # network / HTTP errors
            wait = 2 ** attempt
            log(f"download failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    if ds is None:
        raise RuntimeError("could not download dataset")
    X = ds.data.features.apply(pd.to_numeric, errors="coerce").astype(float)
    y_raw = ds.data.targets.iloc[:, 0].to_numpy().ravel()
    if not set(np.unique(y_raw)).issubset({-1, 1}) or X.isna().any().any():
        raise RuntimeError("unexpected labels or missing feature values")
    # UCI encoding: -1 = phishing (malicious, positive class), 1 = legitimate
    y = (y_raw == -1).astype(int)
    if len(X) > MAX_ROWS:
        X, _, y, _ = train_test_split(X, y, train_size=MAX_ROWS, stratify=y, random_state=SEED)
    return X, y


def evaluate(model, X_te, y_te):
    """Shared evaluation: PR-AUC (main), Brier score and F1 on malicious class (threshold 0.5)."""
    p = model.predict_proba(X_te)[:, 1]
    return {
        "pr_auc": average_precision_score(y_te, p),
        "brier": brier_score_loss(y_te, p),
        "f1": f1_score(y_te, (p >= 0.5).astype(int)),
    }


def main():
    X, y = load_data()
    log(f"loaded {len(X)} rows, {X.shape[1]} features, malicious share={y.mean():.3f}")

    # Single stratified split, fixed seed; test set is touched only at the end
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    log(f"train={len(X_tr)} test={len(X_te)}")

    # Hyper-parameter search on the training set only (cross-validation, PR-AUC)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = {"n_estimators": [200, 400], "learning_rate": [0.05, 0.1],
            "num_leaves": [15, 31], "min_child_samples": [10, 30]}
    search = GridSearchCV(lgb.LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1),
                          grid, scoring="average_precision", cv=cv, n_jobs=1)
    search.fit(X_tr, y_tr)
    best = search.best_params_
    log(f"best LightGBM params (CV PR-AUC={search.best_score_:.4f}): {best}")

    def make_baseline(seed):
        return RandomForestClassifier(random_state=seed)  # default settings

    def make_improved(seed):
        # class-balanced LightGBM with CV-tuned params, isotonic calibration via internal CV
        base = lgb.LGBMClassifier(class_weight="balanced", random_state=seed, verbose=-1, **best)
        return CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)

    results = {}
    for name, make in [("baseline", make_baseline), ("improved", make_improved)]:
        runs = []
        for s in SEEDS:
            m = make(s).fit(X_tr, y_tr)
            r = evaluate(m, X_te, y_te)
            runs.append(r)
            log(f"{name} seed={s}: PR-AUC={r['pr_auc']:.4f} Brier={r['brier']:.4f} F1={r['f1']:.4f}")
        results[name] = {k: np.array([r[k] for r in runs]) for k in runs[0]}

    b, i = results["baseline"], results["improved"]
    b_mean, i_mean = float(b["pr_auc"].mean()), float(i["pr_auc"].mean())
    if not (np.isfinite(b_mean) and np.isfinite(i_mean)):
        log("non-finite metrics; failing")
        sys.exit(1)

    metrics = {
        "baseline_score": b_mean,
        "improved_score": i_mean,
        "improvement": i_mean - b_mean,  # higher is better: improved - baseline
        "baseline_score_std": float(b["pr_auc"].std()),
        "improved_score_std": float(i["pr_auc"].std()),
        "baseline_f1": float(b["f1"].mean()),
        "improved_f1": float(i["f1"].mean()),
        "baseline_brier": float(b["brier"].mean()),
        "improved_brier": float(i["brier"].mean()),
        "brier_improvement": float(b["brier"].mean() - i["brier"].mean()),  # positive = improved better
    }

    # Optional chart: baseline vs improved PR-AUC (mean +/- std over seeds)
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Baseline RF", "Improved LGBM"], [b_mean, i_mean],
           yerr=[metrics["baseline_score_std"], metrics["improved_score_std"]], capsize=6,
           color=["grey", "steelblue"])
    ax.set_ylabel("Test PR-AUC (higher is better)")
    ax.set_title("Phishing dataset: malicious-class PR-AUC")
    fig.tight_layout()
    fig.savefig("figures/pr_auc_comparison.png", dpi=120)
    plt.close(fig)

    out = {
        "question": "Can a class-balanced, calibrated LightGBM model flag malicious entities more reliably "
                    "than a standard random forest on a public cybersecurity dataset, with probability "
                    "outputs analysts can trust?",
        "method": "Stratified train/test split (fixed seed) on UCI Phishing Websites; baseline default "
                  "RandomForestClassifier vs LightGBM with balanced class weights, CV-tuned on training data, "
                  "isotonic calibration via cross-validation. Test set used once. Seeds 42-44; means reported.",
        "dataset": {
            "name": "UCI Phishing Websites (id=327)",
            "source_url": SOURCE_URL,
            "n_rows": int(len(X)),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "PR-AUC (average precision) on held-out test set, malicious class",
        "higher_is_better": True,
        "baseline": "scikit-learn RandomForestClassifier with default settings (random_state=seed)",
        "improved": "LightGBM with class_weight='balanced', hyperparameters chosen by cross-validated PR-AUC "
                    "on the training set, and isotonic calibration of probabilities via cross-validation",
        "metrics": metrics,
        "notes": "Phishing URLs are a proxy for darknet malicious actors, not the paper's data or task. "
                 "Single split; differences of small magnitude may be within split variability. "
                 "F1 uses a fixed 0.5 threshold on each model's probabilities. Hyperparameters were "
                 "searched once with a fixed seed and a small grid for speed. The paper's own results "
                 "are not stated, so no comparison to it is made.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log("wrote results.json and figures/pr_auc_comparison.png")


if __name__ == "__main__":
    main()
