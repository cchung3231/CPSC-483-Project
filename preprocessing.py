"""
Shared preprocessing for the CPSC 483 Diabetes Prevention Model project.

Every team member imports from this file so that all models are trained and
tested on exactly the same split of the data. That makes the model comparison
in the final report fair.

Usage (from any notebook in the project root):

    from preprocessing import load_data, split_data, make_preprocessor

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = make_preprocessor(scale=True)   # scale=False for tree models
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants shared by the whole team
# ---------------------------------------------------------------------------
DATA_PATH = "Datasets/pima-indians-diabetes-resampled.csv"
TARGET = "HasDiabetes"
FEATURES = ["Preg", "Glucose", "BP", "Skin", "Insulin", "BMI", "Pedigree", "Age"]

# A value of 0 is physically impossible for these measurements, so a 0 means
# "not recorded". Pregnancies (Preg) is NOT in this list: 0 pregnancies is real.
ZERO_AS_MISSING = ["Glucose", "BP", "Skin", "Insulin", "BMI"]

TEST_SIZE = 0.20      # 80% train / 20% test
RANDOM_STATE = 42     # fixed seed so everyone gets the identical split


def load_data(path: str = DATA_PATH):
    """Load the CSV and turn medically invalid zeros into NaN (missing).

    Returns
    -------
    X : DataFrame of the 8 predictor features
    y : Series with the 0/1 target
    """
    df = pd.read_csv(path)
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def split_data(X, y, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Stratified train/test split.

    stratify=y keeps the same ~65% / 35% class ratio in both sets, so the test
    set looks like the full dataset.
    """
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def make_preprocessor(scale: bool = True) -> Pipeline:
    """Build the preprocessing steps that go in front of a model.

    - SimpleImputer(strategy="median") fills each missing value with that
      column's median. The median is used instead of the mean because Insulin
      and Skin are strongly right-skewed, and a few very large values would
      pull the mean upward.
    - StandardScaler rescales each feature to mean 0 and standard deviation 1.
      Distance- and weight-based models (KNN, SVM, logistic regression, neural
      networks) need this. Tree models (decision tree, random forest) do not,
      so they can pass scale=False.

    Because these steps live inside a Pipeline, the medians and scaling values
    are learned from the TRAINING data only (inside .fit). That prevents data
    leakage from the test set.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


if __name__ == "__main__":
    # Quick sanity check: python preprocessing.py
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Full data:", X.shape)
    print("Missing values after zero -> NaN:\n", X.isna().sum(), sep="")
    print(f"\nTrain: {X_train.shape}, positive rate {y_train.mean():.3f}")
    print(f"Test:  {X_test.shape}, positive rate {y_test.mean():.3f}")
