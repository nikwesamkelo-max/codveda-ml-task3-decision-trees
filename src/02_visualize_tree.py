"""Step 2 - Visualize the structure of the decision tree."""
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text

TRAIN_PATH = "data/processed/train.csv"
MODEL_PATH = "artifacts/tree_unpruned.joblib"
CHART_PATH = "images/tree_unpruned.png"
RULES_PATH = "outputs/tree_rules_unpruned.txt"
TARGET = "species"


def plot_decision_tree(tree, feature_names, path, title):
    """Draw the tree: each box shows the question, class mix, and majority class."""
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(tree, feature_names=feature_names, class_names=list(tree.classes_),
              filled=True, rounded=True, impurity=True, proportion=False,
              fontsize=11, ax=ax)
    ax.set_title(title, fontsize=22, weight="bold")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    tree = joblib.load(MODEL_PATH)
    feature_names = list(pd.read_csv(TRAIN_PATH).drop(columns=TARGET).columns)

    # Text version: the same tree as readable if/else rules
    rules = export_text(tree, feature_names=feature_names)
    print("Decision rules:\n")
    print(rules)

    plot_decision_tree(tree, feature_names, CHART_PATH,
                       f"Unpruned decision tree (depth {tree.get_depth()}, "
                       f"{tree.get_n_leaves()} leaves)")

    # Checks
    assert rules.count("class:") == tree.get_n_leaves(), "Rules do not match leaves"
    assert os.path.getsize(CHART_PATH) > 0
    print("Visualization verified: one rule per leaf")

    os.makedirs("outputs", exist_ok=True)
    with open(RULES_PATH, "w") as f:
        f.write(rules)
    print(f"Saved -> {CHART_PATH} and {RULES_PATH}")


if __name__ == "__main__":
    main()
