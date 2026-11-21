import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
LOCAL = "data/data_banknote_authentication.txt"
FEATURES = ["variance", "skewness", "curtosis", "entropy"]
COLS = FEATURES + ["class"]
SEED = 42

for folder in ["data", "models", "outputs"]:
    os.makedirs(folder, exist_ok=True)


def load_data():
    """Use the local file if it exists, otherwise download it from UCI (free)."""
    if os.path.exists(LOCAL):
        return pd.read_csv(LOCAL, header=None, names=COLS)
    df = pd.read_csv(URL, header=None, names=COLS)
    df.to_csv(LOCAL, header=False, index=False)
    return df


# 1. LOAD + EXPLORE
df = load_data()
print("Shape:", df.shape)
print(df.info())
print(df.describe().round(3))
print("Missing values:\n", df.isnull().sum())
dupes = df.duplicated().sum()
print("Duplicate rows:", dupes)
df = df.drop_duplicates().reset_index(drop=True)
print("After removing duplicates:", df.shape)
print("Class counts (0 = genuine, 1 = fake):\n", df["class"].value_counts())

df.to_csv("models/reference_data.csv", index=False)

plt.figure(figsize=(5, 4))
sns.countplot(x="class", hue="class", data=df, palette="Set2", legend=False)
plt.xticks([0, 1], ["Genuine", "Fake"])
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig("outputs/01_class_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation")
plt.tight_layout()
plt.savefig("outputs/02_correlation_heatmap.png", dpi=150)
plt.close()

g = sns.pairplot(df, hue="class", palette="Set2", corner=True)
g.savefig("outputs/03_pairplot.png", dpi=110)
plt.close("all")

# 2. SPLIT
X = df[FEATURES]
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# 3. MODELS
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(probability=True, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    "Voting Ensemble (LR+SVM+RF)": VotingClassifier(
        estimators=[("lr", LogisticRegression(max_iter=500)),
                    ("svm", SVC(probability=True, random_state=SEED)),
                    ("rf", RandomForestClassifier(n_estimators=200, random_state=SEED))],
        voting="soft",
    ),
}

rows = []
fitted = {}
for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "cv_f1_mean": cv_f1,
    })
    fitted[name] = pipe

res = pd.DataFrame(rows).round(4).sort_values(["cv_f1_mean", "f1"], ascending=False)
print("\n=== MODEL COMPARISON (fake = positive class) ===")
print(res.to_string(index=False))
res.to_csv("outputs/model_comparison.csv", index=False)

plt.figure(figsize=(8, 4))
res.set_index("model")[["accuracy", "recall", "f1"]].plot(kind="bar", ax=plt.gca())
plt.ylim(0.9, 1.01)
plt.xticks(rotation=25, ha="right")
plt.title("Model Comparison")
plt.tight_layout()
plt.savefig("outputs/04_model_comparison.png", dpi=150)
plt.close()

# 4. BEST MODEL + EVALUATION
best_name = res.iloc[0]["model"]
best = fitted[best_name]
best_pred = best.predict(X_test)
print("\nBest model:", best_name)
report = classification_report(y_test, best_pred, target_names=["Genuine", "Fake"])
print(report)
with open("outputs/classification_report.txt", "w") as f:
    f.write("Best model: " + best_name + "\n\n" + report)

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Genuine", "Fake"], yticklabels=["Genuine", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix\n" + best_name)
plt.tight_layout()
plt.savefig("outputs/05_confusion_matrix.png", dpi=150)
plt.close()

# 5. EXPLAINABILITY
imp = permutation_importance(best, X_test, y_test, scoring="f1",
                             n_repeats=20, random_state=SEED)
imp_df = pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean}) \
    .sort_values("importance", ascending=True)
print("\nPermutation importance:\n", imp_df.sort_values("importance", ascending=False).round(4))

plt.figure(figsize=(6, 3.5))
plt.barh(imp_df["feature"], imp_df["importance"], color="#4682B4")
plt.xlabel("Drop in F1 when feature is shuffled")
plt.title("Feature Importance (Permutation)")
plt.tight_layout()
plt.savefig("outputs/06_feature_importance.png", dpi=150)
plt.close()

# 6. ROBUSTNESS TEST
rng = np.random.default_rng(SEED)
feature_std = X_train.std().values
noise_rows = []
for level in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]:
    noisy = X_test.values + rng.normal(0, 1, X_test.shape) * feature_std * level
    noisy_df = pd.DataFrame(noisy, columns=FEATURES)
    p_noisy = best.predict(noisy_df)
    noise_rows.append({
        "noise_level": level,
        "accuracy": accuracy_score(y_test, p_noisy),
        "fake_recall": recall_score(y_test, p_noisy),
    })
noise_df = pd.DataFrame(noise_rows).round(4)
print("\n=== ROBUSTNESS TO NOISE ===")
print(noise_df.to_string(index=False))
noise_df.to_csv("outputs/robustness.csv", index=False)

plt.figure(figsize=(6, 4))
plt.plot(noise_df["noise_level"], noise_df["accuracy"], marker="o", label="Accuracy")
plt.plot(noise_df["noise_level"], noise_df["fake_recall"], marker="s", label="Fake recall")
plt.xlabel("Noise level (fraction of each feature's std)")
plt.ylabel("Score")
plt.title("Robustness to Noisy Input")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/07_robustness.png", dpi=150)
plt.close()

# 7. SAVE MODEL + EXTRAS FOR THE APP
joblib.dump(best, "models/banknote_model.joblib")

samples = {
    "genuine": X_test[y_test == 0].iloc[0].round(4).to_dict(),
    "fake": X_test[y_test == 1].iloc[0].round(4).to_dict(),
    "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in FEATURES},
    "best_model": best_name,
}
with open("models/samples.json", "w") as f:
    json.dump(samples, f, indent=2)

summary = {
    "best_model": best_name,
    "test_accuracy": float(accuracy_score(y_test, best_pred)),
    "test_precision": float(precision_score(y_test, best_pred)),
    "test_recall": float(recall_score(y_test, best_pred)),
    "test_f1": float(f1_score(y_test, best_pred)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "duplicates_removed": int(dupes),
}
with open("outputs/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nDone. Model saved to models/banknote_model.joblib")
print("Report images and tables are in the outputs/ folder.")
