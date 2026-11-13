"""My Task 1 - Titanic Survival Classification (Arch Technologies, Month 1).

I used the Kaggle Titanic train.csv (891 passengers, 12 columns).
Source: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
(This is a public copy of the Kaggle train.csv file.)

What I did: I loaded the data, looked at it, cleaned it, made new
features, split it 80/20 with seed 42, trained Logistic Regression and
Random Forest, and checked both on the test set.

Run it from the task folder:  python src/titanic.py
It saves: data/titanic_clean.csv, outputs/metrics.json, outputs/*.png
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = pathlib.Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1F4E79"
BLUE = "#2E75B6"
LIGHT = "#5B9BD5"
GREY = "#7F7F7F"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.edgecolor": "#404040", "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 150,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("chart:", name)


def main():
    raw = DATA / "titanic_raw.csv"
    df = pd.read_csv(raw)
    print(f"raw shape: {df.shape}  (expected 891 rows x 12 columns)")
    assert df.shape == (891, 12), f"unexpected shape {df.shape}"

    # ---------- First look at the data ----------
    eda = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "overall_survival_rate": round(float(df["Survived"].mean()), 4),
        "survived": int(df["Survived"].sum()),
        "died": int((1 - df["Survived"]).sum()),
        "missing_age": int(df["Age"].isna().sum()),
        "missing_cabin": int(df["Cabin"].isna().sum()),
        "missing_embarked": int(df["Embarked"].isna().sum()),
        "missing_fare": int(df["Fare"].isna().sum()),
        "survival_by_class": {str(k): round(float(v), 4)
                              for k, v in df.groupby("Pclass")["Survived"].mean().items()},
        "survival_by_sex": {str(k): round(float(v), 4)
                            for k, v in df.groupby("Sex")["Survived"].mean().items()},
        "median_age": round(float(df["Age"].median()), 1),
        "median_fare": round(float(df["Fare"].median()), 2),
    }
    print(json.dumps(eda, indent=2))

    # ---------- Cleaning - I wrote down why I did each step ----------
    # Cabin: 77% missing -> drop the column; it carries almost no signal.
    df = df.drop(columns=["Cabin"])
    # Age: 177 missing (19.9%) -> median Age within each Sex x Pclass group.
    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
        lambda s: s.fillna(s.median()))
    # Embarked: 2 missing -> most common port, Southampton ('S').
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    # Fare: no missing values; median fill kept as a safety net only.
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    assert df.isna().sum().sum() == 0, "missing values remain after cleaning"

    # ---------- Feature engineering ----------
    df["Title"] = df["Name"].str.extract(r",\s*([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
         "Sir", "Jonkheer", "Dona", "Mme", "Ms", "Mlle"], "Rare")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Sex"] = (df["Sex"] == "female").astype(int)  # 1 = female, 0 = male

    features = pd.get_dummies(
        df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
            "FamilySize", "IsAlone", "Embarked", "Title"]],
        columns=["Embarked", "Title"], drop_first=True, dtype=int)
    target = df["Survived"]

    clean = pd.concat([df[["PassengerId", "Survived"]], features], axis=1)
    clean.to_csv(DATA / "titanic_clean.csv", index=False)
    print(f"cleaned shape: {clean.shape}, features: {list(features.columns)}")

    # ---------- Split 80/20, same survival rate in both parts, seed 42 ----------
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42, stratify=target)
    print(f"train {X_train.shape[0]} rows, test {X_test.shape[0]} rows")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---------- Models ----------
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    def evaluate(model, X, name):
        pred = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        return {
            "model": name,
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred)), 4),
            "recall": round(float(recall_score(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        }

    results = {
        "dataset": {"source": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
                    "rows": 891, "columns": 12, "note": "Kaggle Titanic train.csv"},
        "eda": eda,
        "cleaning": {
            "cabin": "dropped (687/891 missing, 77%)",
            "age": "filled with median Age per Sex x Pclass group (177 missing)",
            "embarked": "filled with mode 'S' (2 missing)",
            "fare": "no missing values",
        },
        "split": {"test_size": 0.20, "random_state": 42, "stratified": True,
                  "train_rows": int(X_train.shape[0]), "test_rows": int(X_test.shape[0])},
        "models": [
            evaluate(lr, X_test_s, "Logistic Regression"),
            evaluate(rf, X_test, "Random Forest"),
        ],
        "feature_importance": {
            str(f): round(float(v), 4)
            for f, v in sorted(zip(features.columns, rf.feature_importances_),
                               key=lambda t: -t[1])
        },
        "confusion_matrix_random_forest": confusion_matrix(
            y_test, rf.predict(X_test)).tolist(),
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["models"], indent=2))

    # ---------- Charts ----------
    # 1. Survival rate by passenger class
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cls = df.groupby("Pclass")["Survived"].mean()
    ax.bar([f"Class {i}" for i in cls.index], cls.values, color=[NAVY, BLUE, LIGHT])
    for i, v in enumerate(cls.values):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Survival rate")
    ax.set_title("Survival Rate by Passenger Class")
    save(fig, "survival_by_class.png")

    # 2. Survival rate by sex
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sex = df.groupby("Sex")["Survived"].mean()
    ax.bar(["Male", "Female"], [sex[0], sex[1]], color=[GREY, NAVY])
    for i, v in enumerate([sex[0], sex[1]]):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Survival rate")
    ax.set_title("Survival Rate by Sex")
    save(fig, "survival_by_sex.png")

    # 3. Age distribution by survival
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df.loc[df["Survived"] == 0, "Age"], bins=25, alpha=0.75,
            label="Did not survive", color=GREY)
    ax.hist(df.loc[df["Survived"] == 1, "Age"], bins=25, alpha=0.75,
            label="Survived", color=NAVY)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Passengers")
    ax.set_title("Age Distribution by Survival")
    ax.legend()
    save(fig, "age_distribution.png")

    # 4. Confusion matrix (Random Forest)
    cm = np.array(results["confusion_matrix_random_forest"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: Died", "Predicted: Survived"])
    ax.set_yticklabels(["Actual: Died", "Actual: Survived"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion Matrix — Random Forest (test set)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "confusion_matrix.png")

    # 5. ROC curves (both models)
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, X, label, color in [(lr, X_test_s, "Logistic Regression", BLUE),
                                   (rf, X_test, "Random Forest", NAVY)]:
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X)[:, 1])
        auc = roc_auc_score(y_test, model.predict_proba(X)[:, 1])
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Random guessing")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right")
    save(fig, "roc_curve.png")

    # 6. Feature importance (Random Forest)
    imp = sorted(zip(features.columns, rf.feature_importances_), key=lambda t: -t[1])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([f for f, _ in imp][::-1], [v for _, v in imp][::-1], color=BLUE)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance — Random Forest")
    save(fig, "feature_importance.png")

    print("Task 1 done. Outputs in", OUT)


if __name__ == "__main__":
    main()
