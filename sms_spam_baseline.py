from __future__ import annotations

import io
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, precision_recall_curve
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "SMSSpamCollection"
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def load_dataset() -> pd.DataFrame:
    """Load UCI tab-separated data, downloading it if necessary."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        print("Dataset not found locally. Downloading from UCI...")
        try:
            request = urllib.request.Request(
                DATA_URL, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                archive_bytes = response.read()
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                member = next(
                    (n for n in archive.namelist()
                     if n.replace("\\", "/").endswith("SMSSpamCollection")),
                    None
                )
                if member is None:
                    raise FileNotFoundError(
                        "The downloaded ZIP did not contain SMSSpamCollection."
                    )
                DATA_FILE.write_bytes(archive.read(member))
        except Exception as exc:
            raise RuntimeError(
                "Could not download the UCI dataset. Download "
                "'SMSSpamCollection' from https://archive.ics.uci.edu/dataset/228/"
                "sms+spam+collection and place it at data/SMSSpamCollection, "
                "then run this script again. Original error: " + str(exc)
            ) from exc

    df = pd.read_csv(
        DATA_FILE, sep="\t", header=None, names=["label", "message"],
        encoding="utf-8", encoding_errors="replace", quoting=3
    )
    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str)
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["target"] = (df["label"] == "spam").astype(int)
    return df.reset_index(drop=True)


def normalize_for_grouping(text: str) -> str:
    """Normalize whitespace/case only to keep exact-text duplicates in one fold."""
    return " ".join(str(text).lower().split())


def main() -> None:
    df = load_dataset()
    if len(df) < 20 or df["target"].nunique() != 2:
        raise ValueError("Dataset appears incomplete or does not contain both classes.")

    print("\n=== DATASET EDA ===")
    print(f"Rows loaded: {len(df):,}")
    print(f"Missing messages: {df['message'].isna().sum()}")
    print(f"Exact duplicate message rows: {df['message'].duplicated().sum():,}")
    print("\nClass counts:")
    print(df["label"].value_counts().to_string())
    print("\nClass percentages:")
    print((df["label"].value_counts(normalize=True) * 100).round(2).astype(str).add("%").to_string())

    df["char_length"] = df["message"].str.len()
    print("\nMessage character length by class:")
    print(df.groupby("label")["char_length"].agg(["count", "mean", "median", "std", "min", "max"]).round(2))

    # Group repeated normalized messages so copies do not appear in both partitions.
    # StratifiedGroupKFold attempts to preserve class proportions while respecting groups.
    X = df["message"].astype(str)
    y = df["target"].astype(int)
    groups = df["message"].map(normalize_for_grouping)

    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    overlap = set(groups_train).intersection(set(groups_test))
    if overlap:
        raise RuntimeError("Duplicate-text groups overlap across train/test partitions.")

    print("\n=== SPLIT AUDIT ===")
    print(f"Random seed: {RANDOM_STATE}")
    print(f"Training rows: {len(train_idx):,}")
    print(f"Test rows: {len(test_idx):,}")
    print(f"Train spam rate: {y_train.mean():.4f}")
    print(f"Test spam rate: {y_test.mean():.4f}")
    print(f"Normalized duplicate groups crossing split: {len(overlap)}")

    # Crucial: TF-IDF vocabulary and IDF are learned only when Pipeline.fit()
    # receives X_train. The held-out test text is transformed, never used to fit.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="liblinear"
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Spam precision": precision_score(y_test, y_pred, zero_division=0),
        "Spam recall": recall_score(y_test, y_pred, zero_division=0),
        "Spam F1-score": f1_score(y_test, y_pred, zero_division=0),
        "AUPRC / Average precision": average_precision_score(y_test, y_score),
    }

    print("\n=== HELD-OUT TEST RESULTS (REAL RUN OUTPUT) ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("\nClassification report (0=ham, 1=spam):")
    print(classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["ham", "spam"],
        digits=4, zero_division=0
    ))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    # Cross-validation: every fold fits its own vectorizer inside the pipeline.
    # Groups prevent normalized duplicate texts from crossing folds.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        model, X, y, groups=groups, cv=cv,
        scoring="average_precision", n_jobs=-1
    )
    print("\n=== 5-FOLD GROUPED STRATIFIED CV ===")
    print("AUPRC / Average precision per fold:", np.round(cv_scores, 4))
    print(f"Mean AUPRC: {cv_scores.mean():.4f}")
    print(f"Standard deviation: {cv_scores.std(ddof=1):.4f}")

    # Save auditable outputs.
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    pd.DataFrame([metrics]).to_csv(results_dir / "test_metrics.csv", index=False)
    pd.DataFrame({
        "fold": np.arange(1, len(cv_scores) + 1),
        "average_precision": cv_scores
    }).to_csv(results_dir / "cross_validation_metrics.csv", index=False)

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=[0, 1], display_labels=["ham", "spam"],
        values_format="d"
    )
    plt.title("SMS Spam Detection — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall (spam)")
    plt.ylabel("Precision (spam)")
    plt.title(f"Precision–Recall Curve (AP={metrics['AUPRC / Average precision']:.3f})")
    plt.tight_layout()
    plt.savefig(results_dir / "precision_recall_curve.png", dpi=180)
    plt.close()

    print("\nSaved: results/test_metrics.csv")
    print("Saved: results/cross_validation_metrics.csv")
    print("Saved: results/confusion_matrix.png")
    print("Saved: results/precision_recall_curve.png")
    print("\nReproducibility note: same dataset version, Python/library versions, and seed")
    print("are needed for exact repeatability. The split is group-aware; because this")
    print("can slightly change class proportions, report the observed split rates above.")


if __name__ == "__main__":
    main()
