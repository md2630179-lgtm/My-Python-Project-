"""
ML মডেল ইভ্যালুয়েশন প্লট:
1. Feature Importance Plot
2. Confusion Matrix
3. ROC-AUC Curve
4. Distribution Graph

প্রয়োজনীয় লাইব্রেরি:
pip install numpy pandas matplotlib seaborn scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    classification_report,
)

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------
# 1. Feature Importance Plot
# ---------------------------------------------------------------
def plot_feature_importance(model, feature_names, top_n=15, save_path=None):
    importances = model.feature_importances_
    df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(9, 6))
    sns.barplot(data=df, x="importance", y="feature", hue="feature",
                palette="viridis", legend=False)
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ---------------------------------------------------------------
# 2. Confusion Matrix
# ---------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=None,
                          save_path=None):
    """normalize: None | 'true' | 'pred' | 'all'"""
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", ax=ax, values_format=".2f" if normalize else "d")
    ax.set_title("Confusion Matrix" + (f" (normalize={normalize})" if normalize else ""))
    ax.grid(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ---------------------------------------------------------------
# 3. ROC-AUC Curve
# ---------------------------------------------------------------
def plot_roc_curve(y_true, y_proba, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return auc


# ---------------------------------------------------------------
# 4. Distribution Graphs
# ---------------------------------------------------------------
def plot_feature_distributions(df, columns, hue=None, cols_per_row=3, save_path=None):
    """নির্বাচিত ফিচারগুলোর হিস্টোগ্রাম + KDE (ঐচ্ছিকভাবে ক্লাস অনুযায়ী)"""
    n = len(columns)
    rows = int(np.ceil(n / cols_per_row))
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 4 * rows))
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, columns):
        sns.histplot(data=df, x=col, hue=hue, kde=True, ax=ax, palette="Set2")
        ax.set_title(f"Distribution of {col}")

    for ax in axes[n:]:  # অতিরিক্ত খালি সাবপ্লট বন্ধ
        ax.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_target_distribution(y, class_names=None, save_path=None):
    """টার্গেট ক্লাসের ডিস্ট্রিবিউশন (class imbalance দেখার জন্য)"""
    counts = pd.Series(y).value_counts().sort_index()
    labels = class_names if class_names is not None else counts.index

    plt.figure(figsize=(5, 4))
    ax = sns.barplot(x=list(labels), y=counts.values, hue=list(labels),
                     palette="pastel", legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center")
    plt.title("Target Class Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_predicted_probability_distribution(y_true, y_proba, save_path=None):
    """প্রতিটি ক্লাসের জন্য predicted probability-র ডিস্ট্রিবিউশন"""
    df = pd.DataFrame({"proba": y_proba, "actual": y_true})
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="proba", hue="actual", bins=30, kde=True,
                 palette="Set1", element="step")
    plt.title("Predicted Probability Distribution by Actual Class")
    plt.xlabel("Predicted probability (class 1)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ---------------------------------------------------------------
# Demo: উদাহরণ ডেটাসেট (Breast Cancer) দিয়ে চালানো
# ---------------------------------------------------------------
if __name__ == "__main__":
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    class_names = list(data.target_names)  # ['malignant', 'benign']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # positive class (1) এর probability

    print(classification_report(y_test, y_pred, target_names=class_names))

    # 1. Feature importance
    plot_feature_importance(model, X.columns, top_n=15,
                            save_path="feature_importance.png")

    # 2. Confusion matrix (raw count ও normalized)
    plot_confusion_matrix(y_test, y_pred, class_names, save_path="confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, class_names, normalize="true",
                          save_path="confusion_matrix_normalized.png")

    # 3. ROC-AUC
    auc = plot_roc_curve(y_test, y_proba, save_path="roc_curve.png")
    print(f"ROC-AUC: {auc:.4f}")

    # 4. Distribution graphs
    df_plot = X.copy()
    df_plot["target"] = y.map(dict(enumerate(class_names)))
    plot_feature_distributions(
        df_plot,
        columns=["mean radius", "mean texture", "mean area"],
        hue="target",
        save_path="feature_distributions.png",
    )
    plot_target_distribution(y, class_names, save_path="target_distribution.png")
    plot_predicted_probability_distribution(y_test, y_proba,
                                            save_path="proba_distribution.png")
