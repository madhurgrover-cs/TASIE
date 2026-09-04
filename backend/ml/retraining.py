"""
Retraining pipeline — run this weekly (or manually):
    python -m backend.ml.retraining

Reads unlabelled Feedback rows where developer_label has been set but
is_used_for_training=False, trains a TF-IDF + LogisticRegression model, and
saves the new version to the model registry.
"""
from collections import Counter

from sqlalchemy.orm import Session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from backend.database import SessionLocal
from backend.models import Feedback, FeedbackLabel
from backend.ml.model_registry import save_model

# Map labels to binary training signal
LABEL_MAP = {
    FeedbackLabel.VALID_VULNERABILITY: "valid_vulnerability",
    FeedbackLabel.FALSE_POSITIVE: "false_positive",
    FeedbackLabel.NEEDS_REVIEW: "needs_review",
}

MIN_SAMPLES = 10  # Minimum labelled samples required before training


def train_model():
    db: Session = SessionLocal()
    try:
        rows = (
            db.query(Feedback)
            .filter(
                Feedback.developer_label.isnot(None),
                Feedback.is_used_for_training == False,
            )
            .all()
        )

        if len(rows) < MIN_SAMPLES:
            print(f"[Retraining] Not enough samples ({len(rows)}/{MIN_SAMPLES}). Skipping.")
            return None

        X = [r.code_snippet for r in rows]
        y = [LABEL_MAP[r.developer_label] for r in rows]

        # A classifier needs at least two distinct labels to train.
        class_counts = Counter(y)
        if len(class_counts) < 2:
            print(f"[Retraining] Only one label present ({list(class_counts)}). "
                  "Need at least 2 distinct labels. Skipping.")
            return None

        # Feature extraction
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_vec = vectorizer.fit_transform(X)

        # Hold out a stratified test set only when every class has enough samples
        # to land on both sides of the split; otherwise fall back to evaluating
        # on the training data (small-dataset PoC behaviour).
        can_split = min(class_counts.values()) >= 2 and len(rows) >= 10
        if can_split:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_vec, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError:
                X_train, X_test, y_train, y_test = X_vec, X_vec, y, y
        else:
            print("[Retraining] Dataset too small/imbalanced for a hold-out split; "
                  "evaluating on training data.")
            X_train, X_test, y_train, y_test = X_vec, X_vec, y, y

        # Train
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        print("[Retraining] Evaluation:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Calculate false positive reduction threshold (conservative, max 0.5)
        if "false_positive" in report:
            fp_precision = report["false_positive"].get("precision", 0.5)
            threshold = min(0.5, max(0.3, 1.0 - fp_precision))
        else:
            threshold = 0.5

        metrics = {
            "samples_trained": len(rows),
            "threshold": threshold,
            "report": {k: v for k, v in report.items() if k != "accuracy"},
        }

        version = save_model(vectorizer, clf, threshold, metrics)
        print(f"[Retraining] Model saved as {version}")

        # Mark rows as used
        for row in rows:
            row.is_used_for_training = True
        db.commit()

        return version

    finally:
        db.close()


if __name__ == "__main__":
    train_model()
