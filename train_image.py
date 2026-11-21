"""train_image.py - trains the PHOTO model from folders of real and fake note photos.
Usage:  python train_image.py --name "Dataset name"
Folders: image_data/genuine/  and  image_data/fake/  (sub-folders are fine;
'real' / 'counterfeit' folder names also work)."""
import os, sys, json, argparse, random, datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, confusion_matrix, classification_report)
import imgfeat

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="image_data")
ap.add_argument("--name", default="Custom banknote photo dataset")
ap.add_argument("--max", type=int, default=1200, help="max images used per class")
args = ap.parse_args()

EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
ALIASES = {0: ("genuine", "real", "original"), 1: ("fake", "counterfeit", "forged", "forgery")}
SEED = 42
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

if not os.path.isdir(args.data):
    sys.exit("Folder not found: " + args.data)


def files_for(label):
    out = []
    for sub in os.listdir(args.data):
        if sub.lower() in ALIASES[label]:
            for root, _, fs in os.walk(os.path.join(args.data, sub)):
                out += [os.path.join(root, f) for f in fs if f.lower().endswith(EXT)]
    return out


random.seed(SEED)
X, y = [], []
for label in (0, 1):
    files = files_for(label)
    print("Found", len(files), "images for", "genuine" if label == 0 else "fake")
    if len(files) > args.max:
        files = random.sample(files, args.max)
    for i, p in enumerate(files):
        try:
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            vec, _ = imgfeat.extract(im)
            X.append(vec)
            y.append(label)
        except Exception as e:
            print("  skipped", os.path.basename(p), type(e).__name__)
        if (i + 1) % 100 == 0:
            print("  processed", i + 1, "of", len(files))

X, y = np.array(X), np.array(y)
n0, n1 = int((y == 0).sum()), int((y == 1).sum())
print("Using", n0, "genuine and", n1, "fake images")
if min(n0, n1) < 10:
    sys.exit("Need at least 10 images in each class (genuine and fake).")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
cv = StratifiedKFold(n_splits=min(5, int(min((ytr == 0).sum(), (ytr == 1).sum()))), shuffle=True, random_state=SEED)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "SVM (RBF)": SVC(probability=True, class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
    "Extra Trees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
}
rows, fitted = [], {}
for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    cvf = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1").mean()
    pipe.fit(Xtr, ytr)
    pr = pipe.predict(Xte)
    rows.append({"model": name, "accuracy": accuracy_score(yte, pr), "precision": precision_score(yte, pr, zero_division=0),
                 "recall": recall_score(yte, pr), "f1": f1_score(yte, pr),
                 "balanced_accuracy": balanced_accuracy_score(yte, pr), "cv_f1_mean": cvf})
    fitted[name] = pipe

res = pd.DataFrame(rows).round(4).sort_values(["cv_f1_mean", "balanced_accuracy"], ascending=False)
print("\n=== PHOTO MODEL COMPARISON (fake = positive class) ===")
print(res.to_string(index=False))
res.to_csv("outputs/image_model_comparison.csv", index=False)

best_name = res.iloc[0]["model"]
best = fitted[best_name]
pred = best.predict(Xte)
print("\nBest model:", best_name)
print(classification_report(yte, pred, target_names=["Genuine", "Fake"], zero_division=0))

cm = confusion_matrix(yte, pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Greens")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
plt.xticks([0, 1], ["Genuine", "Fake"])
plt.yticks([0, 1], ["Genuine", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Photo model confusion matrix\n" + best_name)
plt.tight_layout()
plt.savefig("outputs/08_photo_confusion_matrix.png", dpi=150)
plt.close()

joblib.dump(best, "models/image_model.joblib")
info = {
    "best_model": best_name, "dataset_name": args.name,
    "accuracy": float(accuracy_score(yte, pred)), "precision": float(precision_score(yte, pred, zero_division=0)),
    "recall": float(recall_score(yte, pred)), "f1": float(f1_score(yte, pred)),
    "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
    "n_total": int(len(y)), "n_genuine": n0, "n_fake": n1, "n_test": int(len(yte)),
    "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
}
with open("models/image_summary.json", "w") as f:
    json.dump(info, f, indent=2)
print("\nSaved models/image_model.joblib. Restart the website (python app.py) to use it.")
