"""Step 1 - Train a decision tree classifier on the Iris dataset."""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RAW_PATH = "data/raw/iris.csv"
OUT_DIR = "data/processed"
MODEL_PATH = "artifacts/tree_unpruned.joblib"
TARGET = "species"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path=RAW_PATH):
    return pd.read_csv(path)


def split_data(df):
    """80/20 split. 'stratify' keeps the three species in equal proportion."""
    X, y = df.drop(columns=TARGET), df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)


def train_tree(X_train, y_train):
    """No limits on growth: the tree keeps splitting until every leaf is pure."""
    return DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)


def main():
    df = load_data()
    print(f"Loaded {df.shape[0]} flowers, {df.shape[1] - 1} features")
    print(f"Missing values: {int(df.isna().sum().sum())} | Classes: {sorted(df[TARGET].unique())}\n")

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train)} | Test: {len(X_test)} (test per class: "
          f"{y_test.value_counts().to_dict()})\n")

    tree = train_tree(X_train, y_train)
    t = tree.tree_
    root = X_train.columns[t.feature[0]]
    print(f"Tree depth: {tree.get_depth()} | Leaves: {tree.get_n_leaves()} | Nodes: {t.node_count}")
    print(f"First question the tree asks: is {root} <= {t.threshold[0]:.2f}?")
    print(f"Training accuracy: {tree.score(X_train, y_train):.3f}\n")

    print("Feature importance:")
    for name, imp in sorted(zip(X_train.columns, tree.feature_importances_),
                            key=lambda p: -p[1]):
        print(f"  {name:<14}{imp:.3f}")

    # Checks
    assert len(X_train) + len(X_test) == len(df)
    assert set(X_train.index).isdisjoint(X_test.index), "Train and test overlap"
    assert set(tree.classes_) == set(df[TARGET].unique())
    assert abs(tree.feature_importances_.sum() - 1) < 1e-9
    print("\nTraining verified. Test data untouched until step 4")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    X_train.assign(**{TARGET: y_train}).to_csv(f"{OUT_DIR}/train.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(f"{OUT_DIR}/test.csv", index=False)
    joblib.dump(tree, MODEL_PATH)
    print(f"Saved -> {OUT_DIR}/train.csv, test.csv and {MODEL_PATH}")


if __name__ == "__main__":
    main()
