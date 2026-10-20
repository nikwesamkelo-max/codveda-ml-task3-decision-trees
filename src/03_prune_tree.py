"""Step 3 - Prune the decision tree to prevent overfitting."""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

TRAIN_PATH = "data/processed/train.csv"
UNPRUNED_PATH = "artifacts/tree_unpruned.joblib"
PRUNED_PATH = "artifacts/tree_pruned.joblib"
RESULTS_PATH = "outputs/pruning_results.csv"
CURVE_PATH = "images/pruning_curve.png"
TREE_PATH = "images/tree_pruned.png"
TARGET = "species"
RANDOM_STATE = 42


def score_candidates(X, y):
    """Try every pruning strength (alpha) and cross-validate each one.

    Cost-complexity pruning removes the branches that add the least accuracy
    per extra leaf. A larger alpha means a smaller tree. Only TRAINING data
    is used here, so the test set stays untouched.
    """
    path = DecisionTreeClassifier(random_state=RANDOM_STATE).cost_complexity_pruning_path(X, y)
    alphas = path.ccp_alphas[:-1]  # the last alpha would prune the tree to a single node
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    rows = []
    for alpha in alphas:
        tree = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=alpha)
        scores = cross_val_score(tree, X, y, cv=cv)
        tree.fit(X, y)
        rows.append({"alpha": alpha, "leaves": tree.get_n_leaves(), "depth": tree.get_depth(),
                     "train_acc": tree.score(X, y), "cv_acc": scores.mean(),
                     "cv_se": scores.std() / np.sqrt(len(scores))})
    return pd.DataFrame(rows)


def choose_alpha(results):
    """One-standard-error rule: the simplest tree that is within one standard
    error of the best cross-validated accuracy."""
    best = results.loc[results["cv_acc"].idxmax()]
    good_enough = results[results["cv_acc"] >= best["cv_acc"] - best["cv_se"]]
    return good_enough.loc[good_enough["alpha"].idxmax()]


def plot_curve(results, chosen):
    r = results.sort_values("leaves")
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.plot(r["leaves"], r["train_acc"], "o-", color="#f97316", lw=2.5, ms=9, label="Training accuracy")
    ax.errorbar(r["leaves"], r["cv_acc"], yerr=r["cv_se"], fmt="o-", color="#2563eb",
                lw=2.5, ms=9, capsize=5, label="Cross-validated accuracy")
    ax.axvline(chosen["leaves"], color="#16a34a", ls="--", lw=2)
    ax.text(chosen["leaves"] + 0.1, r["cv_acc"].min() + 0.01,
            f"chosen: {int(chosen['leaves'])} leaves", color="#16a34a", fontsize=14, weight="bold")
    ax.set_xlabel("Number of leaves (tree size)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pruning: a smaller tree generalizes as well as the full one",
                 fontsize=19, weight="bold", loc="left")
    ax.legend(fontsize=14, frameon=False, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    os.makedirs("images", exist_ok=True)
    fig.savefig(CURVE_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pruned_tree(tree, feature_names):
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(tree, feature_names=feature_names, class_names=list(tree.classes_),
              filled=True, rounded=True, fontsize=13, ax=ax)
    ax.set_title(f"Pruned decision tree (depth {tree.get_depth()}, {tree.get_n_leaves()} leaves)",
                 fontsize=22, weight="bold")
    fig.savefig(TREE_PATH, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    train = pd.read_csv(TRAIN_PATH)
    X, y = train.drop(columns=TARGET), train[TARGET]
    unpruned = joblib.load(UNPRUNED_PATH)

    results = score_candidates(X, y)
    chosen = choose_alpha(results)
    pruned = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=chosen["alpha"]).fit(X, y)

    print(results.round(4).to_string(index=False), "\n")
    print(f"Chosen alpha: {chosen['alpha']:.4f} (simplest tree within 1 std error of the best)\n")

    base = results.loc[results["alpha"] == 0].iloc[0]
    print(f"{'':<10}{'Depth':>7}{'Leaves':>8}{'Train acc':>11}{'CV acc':>9}")
    print(f"{'Unpruned':<10}{unpruned.get_depth():>7}{unpruned.get_n_leaves():>8}"
          f"{base['train_acc']:>11.3f}{base['cv_acc']:>9.3f}")
    print(f"{'Pruned':<10}{pruned.get_depth():>7}{pruned.get_n_leaves():>8}"
          f"{pruned.score(X, y):>11.3f}{chosen['cv_acc']:>9.3f}")

    # Checks
    best = results["cv_acc"].max()
    assert pruned.get_n_leaves() < unpruned.get_n_leaves(), "Tree was not pruned"
    assert pruned.get_n_leaves() >= len(pruned.classes_), "Tree too small for 3 classes"
    assert chosen["cv_acc"] >= best - results.loc[results["cv_acc"].idxmax(), "cv_se"]
    print("\nPruning verified. Test data untouched until step 4")

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(pruned, PRUNED_PATH)
    results.round(4).to_csv(RESULTS_PATH, index=False)
    plot_curve(results, chosen)
    plot_pruned_tree(pruned, list(X.columns))
    print(f"Saved -> {PRUNED_PATH}, {RESULTS_PATH}, {CURVE_PATH}, {TREE_PATH}")


if __name__ == "__main__":
    main()
