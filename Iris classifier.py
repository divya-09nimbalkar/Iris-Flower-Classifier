# ============================================================
# Iris Flower Classifier — Multi-Model Comparison
# Author  : Divya Nimbalkar
# GitHub  : https://github.com/divya-09nimbalkar
# Dataset : Iris (sklearn built-in)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── 1. Load Data ────────────────────────────────────────────
print("=" * 55)
print("   IRIS FLOWER CLASSIFIER - Multi-Model Comparison")
print("=" * 55)

iris = load_iris()
df   = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species_name"] = df["species"].map(
    {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
)

print(f"\n Dataset shape  : {df.shape}")
print(f" Classes        : {list(iris.target_names)}")
print(f"\n Class distribution:\n{df['species_name'].value_counts()}")
print(f"\n First 5 rows:\n{df.head()}")
print(f"\n Statistics:\n{df.describe().round(2)}")

# ── 2. Visualize ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
features  = iris.feature_names
colors    = ["#e74c3c", "#2ecc71", "#3498db"]
species   = [0, 1, 2]
names     = ["Setosa", "Versicolor", "Virginica"]

for idx, (i, j) in enumerate([(0,1),(0,2),(0,3),(2,3)]):
    ax = axes[idx // 2][idx % 2]
    for sp, color, name in zip(species, colors, names):
        mask = df["species"] == sp
        ax.scatter(df.loc[mask, features[i]],
                   df.loc[mask, features[j]],
                   c=color, label=name, alpha=0.7, s=50)
    ax.set_xlabel(features[i])
    ax.set_ylabel(features[j])
    ax.legend(fontsize=8)

plt.suptitle("Iris Feature Scatter Plots", fontsize=14)
plt.tight_layout()
plt.savefig("iris_scatter.png", dpi=150)
plt.close()
print("\n Saved: iris_scatter.png")

# ── 3. Preprocessing ─────────────────────────────────────────
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)

# -- 4. Train & Compare Models ---------------------------------------
models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree"      : DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)"          : SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
}

print("\n-- Model Comparison --")
print(f"{'Model':<25} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>10}")
print("-" * 60)

best_acc = 0
best_name = ""
all_results = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    acc     = accuracy_score(y_test, y_pred)
    cv      = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"{name:<25} {acc*100:>9.1f}% {cv.mean()*100:>9.1f}% {cv.std()*100:>9.1f}%")
    all_results[name] = {"acc": acc, "cv_mean": cv.mean(), "model": model, "y_pred": y_pred}
    if acc > best_acc:
        best_acc, best_name = acc, name

print(f"\n Best model: {best_name} ({best_acc*100:.1f}% accuracy)")

# -- 5. Best Model - Detailed Report ----------------------------------
best = all_results[best_name]
print(f"\n-- Classification Report ({best_name}) --")
print(classification_report(y_test, best["y_pred"], target_names=iris.target_names))

# ── 6. Confusion Matrix ──────────────────────────────────────
cm = confusion_matrix(y_test, best["y_pred"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# ── 7. Feature Importance (Random Forest) ───────────────────
rf = all_results["Random Forest"]["model"]
imp = pd.DataFrame({
    "Feature"   : iris.feature_names,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n-- Feature Importance (Random Forest) --")
print(imp.to_string(index=False))

plt.figure(figsize=(7, 4))
sns.barplot(data=imp, x="Importance", y="Feature", palette="viridis")
plt.title("Feature Importance — Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("Saved: feature_importance.png")

# -- 8. Predict New Flower -------------------------------------------
print("\n-- Predict New Sample --")
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # typical Setosa
sample_sc = scaler.transform(sample)
pred = best["model"].predict(sample_sc)[0]
print(f" Input: sepal_len=5.1, sepal_wid=3.5, petal_len=1.4, petal_wid=0.2")
print(f" Predicted species: {iris.target_names[pred]}")

print("\n Done!")