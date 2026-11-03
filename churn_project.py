"""
Telecom Customer Churn Prediction  -  একটি end-to-end Data Science project
পরিস্থিতি: বাংলাদেশের একটি mobile operator জানতে চায় কোন গ্রাহক আগামী মাসে
সংযোগ ছেড়ে দিতে পারে (churn), যাতে আগেই তাদের অফার দিয়ে ধরে রাখা যায়।
"""

# ============================================================
# STEP 0: Library import
# ============================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # server-এ chart ফাইলে save করার জন্য
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# ============================================================
# STEP 1: ডেটা তৈরি (real কোম্পানির ডেটা private, তাই synthetic ডেটা)
# ============================================================
rng = np.random.default_rng(42)     # seed দিলে প্রতিবার একই ফল আসে
N = 5000

df = pd.DataFrame({
    "customer_id": np.arange(1, N + 1),
    "age": rng.integers(18, 66, N),
    "division": rng.choice(
        ["Dhaka", "Chattogram", "Rajshahi", "Khulna",
         "Sylhet", "Barishal", "Rangpur", "Mymensingh"],
        N, p=[.30, .18, .10, .10, .08, .06, .09, .09]),
    "plan_type": rng.choice(["Prepaid", "Postpaid"], N, p=[.75, .25]),
    "tenure_months": rng.integers(1, 73, N),
    "monthly_bill": np.clip(rng.normal(550, 200, N), 100, 1500).round(0),
    "data_usage_gb": np.clip(rng.gamma(2.0, 4.0, N), 0.1, 60).round(1),
    "call_minutes": np.clip(rng.normal(300, 120, N), 0, 900).round(0),
    "complaints_last_3m": rng.poisson(0.8, N),
    "late_payments": rng.poisson(0.5, N),
})

# churn কীভাবে ঘটবে তার "গোপন নিয়ম" (বাস্তবে এটা আমরা জানি না, model শিখে বের করবে)
score = (-1.2
         + 0.55 * df["complaints_last_3m"]
         + 0.45 * df["late_payments"]
         - 0.035 * df["tenure_months"]
         + 0.0025 * (df["monthly_bill"] - 550)
         - 0.03 * df["data_usage_gb"]
         + 0.5 * (df["plan_type"] == "Prepaid"))
prob = 1 / (1 + np.exp(-score))
df["churn"] = (rng.random(N) < prob).astype(int)

# বাস্তবের মতো ডেটা নোংরা করা: কিছু মান হারিয়ে গেছে
df.loc[rng.choice(N, 200, replace=False), "data_usage_gb"] = np.nan
df.loc[rng.choice(N, 120, replace=False), "age"] = np.nan

print("=== STEP 1: ডেটার প্রথম ৫ সারি ===")
print(df.head())
print("\nআকার (সারি, কলাম):", df.shape)

# ============================================================
# STEP 2: EDA (Exploratory Data Analysis)
# ============================================================
print("\n=== STEP 2: EDA ===")
print("\nMissing value:")
print(df.isna().sum()[df.isna().sum() > 0])

print("\nChurn হার (%):")
print((df["churn"].value_counts(normalize=True) * 100).round(1))

print("\nDescribe:")
print(df[["tenure_months", "monthly_bill", "data_usage_gb",
          "complaints_last_3m"]].describe().round(1))

print("\nComplaint সংখ্যা অনুযায়ী churn হার (%):")
print((df.groupby("complaints_last_3m")["churn"].mean() * 100).round(1).head(6))

print("\nPlan type অনুযায়ী churn হার (%):")
print((df.groupby("plan_type")["churn"].mean() * 100).round(1))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
(df.groupby("complaints_last_3m")["churn"].mean().head(6) * 100).plot.bar(ax=ax[0], color="#d9534f")
ax[0].set_title("Complaints vs Churn %")
ax[0].set_ylabel("Churn %")
df.boxplot(column="tenure_months", by="churn", ax=ax[1])
ax[1].set_title("Tenure vs Churn"); plt.suptitle("")
(df.groupby("plan_type")["churn"].mean() * 100).plot.bar(ax=ax[2], color="#5b9bd5")
ax[2].set_title("Plan type vs Churn %")
plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/eda.png", dpi=110); plt.close()

# ============================================================
# STEP 3: Feature engineering
# ============================================================
df["bill_per_gb"] = df["monthly_bill"] / (df["data_usage_gb"] + 1)
df["is_new_customer"] = (df["tenure_months"] <= 6).astype(int)
df["issue_score"] = df["complaints_last_3m"] + df["late_payments"]

# ============================================================
# STEP 4: Train / Test split
# ============================================================
X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("\n=== STEP 4 ===")
print("Train:", X_train.shape, "| Test:", X_test.shape)

# ============================================================
# STEP 5: Preprocessing pipeline
# ============================================================
num_cols = ["age", "tenure_months", "monthly_bill", "data_usage_gb",
            "call_minutes", "complaints_last_3m", "late_payments",
            "bill_per_gb", "is_new_customer", "issue_score"]
cat_cols = ["division", "plan_type"]

num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])
cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocess = ColumnTransformer([("num", num_pipe, num_cols),
                                ("cat", cat_pipe, cat_cols)])

# ============================================================
# STEP 6: Model training (baseline + advanced)
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=10, random_state=42, n_jobs=-1),
}

fitted = {}
print("\n=== STEP 6 & 7: Model ও মূল্যায়ন ===")
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    fitted[name] = (pipe, proba)
    print(f"\n{name}  ->  ROC-AUC = {auc:.3f}")

# ============================================================
# STEP 7: সেরা model-এর বিস্তারিত মূল্যায়ন
# ============================================================
best_name = "Logistic Regression"
best_pipe, best_proba = fitted[best_name]

pred = (best_proba >= 0.5).astype(int)
print(f"\n--- {best_name}: threshold = 0.5 ---")
print(classification_report(y_test, pred, target_names=["Stay", "Churn"]))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

# ============================================================
# STEP 8: কোন বিষয়গুলো churn-এর কারণ? (feature importance)
# ============================================================
feature_names = best_pipe.named_steps["prep"].get_feature_names_out()
coefs = best_pipe.named_steps["model"].coef_[0]
imp = (pd.Series(coefs, index=feature_names)
       .sort_values(key=abs, ascending=False).head(8))
print("\n=== STEP 8: সবচেয়ে প্রভাবশালী feature (coefficient) ===")
print(imp.round(3))

# ============================================================
# STEP 9: Business impact ও threshold নির্বাচন
# ============================================================
# ধরি: একজন গ্রাহককে retain অফার দিতে খরচ 100 টাকা,
#      সফল হলে গ্রাহক থেকে বছরে গড়ে 3000 টাকা আয় বাঁচে,
#      অফার দিলে churn-প্রবণ গ্রাহকের 40% থেকে যায়।
COST, VALUE, SUCCESS = 100, 3000, 0.40
print("\n=== STEP 9: Threshold অনুযায়ী মুনাফা (test set, টাকায়) ===")
rows = []
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    flagged = best_proba >= t
    n_flagged = flagged.sum()
    true_churners = (flagged & (y_test.values == 1)).sum()
    profit = true_churners * SUCCESS * VALUE - n_flagged * COST
    rows.append((t, n_flagged, true_churners, profit))
print(pd.DataFrame(rows, columns=["threshold", "অফার দেওয়া হলো",
                                  "সত্যিকারের churner", "মোট মুনাফা"]).to_string(index=False))

# ---- ফলাফলের chart ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name, (_, p) in fitted.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    ax[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", label="Random guess")
ax[0].set_xlabel("False Positive Rate"); ax[0].set_ylabel("True Positive Rate")
ax[0].set_title("ROC Curve"); ax[0].legend()
imp.sort_values().plot.barh(ax=ax[1], color=["#d9534f" if v > 0 else "#5cb85c" for v in imp.sort_values()])
ax[1].set_title("Feature effect (red = raises churn)")
plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/results.png", dpi=110); plt.close()

# ---- ভবিষ্যতে ব্যবহারের জন্য একজন নতুন গ্রাহকের ভবিষ্যদ্বাণী ----
new_customer = pd.DataFrame([{
    "age": 27, "division": "Dhaka", "plan_type": "Prepaid",
    "tenure_months": 4, "monthly_bill": 700, "data_usage_gb": 3.0,
    "call_minutes": 150, "complaints_last_3m": 3, "late_payments": 2,
    "bill_per_gb": 700 / 4.0, "is_new_customer": 1, "issue_score": 5}])
p = best_pipe.predict_proba(new_customer)[0, 1]
print(f"\nনতুন গ্রাহকের churn সম্ভাবনা: {p*100:.1f}%")
