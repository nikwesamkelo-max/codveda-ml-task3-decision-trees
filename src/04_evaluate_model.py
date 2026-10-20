"""Step 4 - Evaluate the unpruned and pruned trees with accuracy and F1-score."""
import os
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             ConfusionMatrixDisplay)

TEST_PATH = "data/processed/test.csv"
UNPRUNED_PATH = "artifacts/tree_unpruned.joblib"
PRUNED_PATH = "artifacts/tree_pruned.joblib"
METRICS_PATH = "outputs/metrics.json"
REPORT_PATH = "outputs/classification_report_pruned.txt"
CHART_PATH = "images/confusion_matrices.png"
TARGET = "species"


def compute_metrics(y_true, y_pred):
    """Accuracy plus F1 (macro = every species counts equally)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def plot_confusion_matrices(y_true, predictions, classes):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=classes, cmap="Blues", colorbar=False, ax=ax)
        acc = accuracy_score(y_true, y_pred)
        ax.set_title(f"{name} (accuracy {acc:.1%})", fontsize=16, weight="bold")
        ax.set_xticklabels(classes, rotation=20)
    axes[1].set_ylabel("")  # avoid overlapping the left labels
    fig.suptitle("Predictions on 30 unseen flowers", fontsize=20, weight="bold")
    os.makedirs("images", exist_ok=True)
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    test = pd.read_csv(TEST_PATH)
    X_test, y_test = test.drop(columns=TARGET), test[TARGET]
    models = {"Unpruned": joblib.load(UNPRUNED_PATH), "Pruned": joblib.load(PRUNED_PATH)}

    predictions = {name: m.predict(X_test) for name, m in models.items()}
    metrics = {name: compute_metrics(y_test, p) for name, p in predictions.items()}

    print(f"Evaluated on {len(y_test)} unseen flowers\n")
    print(f"{'':<10}{'Accuracy':>10}{'F1 macro':>10}{'F1 weighted':>13}{'Leaves':>8}")
    for name, m in metrics.items():
        print(f"{name:<10}{m['accuracy']:>10.3f}{m['f1_macro']:>10.3f}"
              f"{m['f1_weighted']:>13.3f}{models[name].get_n_leaves():>8}")

    report = classification_report(y_test, predictions["Pruned"], digits=3)
    print("\nPruned tree, per species:\n" + report)
    wrong = int((predictions["Pruned"] != y_test).sum())
    print(f"Pruned tree mistakes: {wrong} of {len(y_test)} flowers "
          f"(each flower is worth {1 / len(y_test):.1%} accuracy)")

    # Checks
    assert all(len(p) == len(y_test) for p in predictions.values())
    assert 0 <= metrics["Pruned"]["accuracy"] <= 1
    assert abs(metrics["Pruned"]["accuracy"] - models["Pruned"].score(X_test, y_test)) < 1e-12
    print("\nEvaluation verified")

    os.makedirs("outputs", exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    plot_confusion_matrices(y_test, predictions, list(models["Pruned"].classes_))
    print(f"Saved -> {METRICS_PATH}, {REPORT_PATH}, {CHART_PATH}")


if __name__ == "__main__":
    main()
