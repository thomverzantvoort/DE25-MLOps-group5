# training-api/train.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# --- 1. Load Data ---
DATA_PATH = r"C:\Users\rderi\OneDrive\Bureaublad\Jads 2025-2027\Data Engineering\Data Engineering Assignment\DE25-MLOps-group5\Data\spotify_churn_dataset.csv"
df_raw = pd.read_csv(DATA_PATH)
print(f"✅ Data loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# --- 2. Remove redundant / unnecessary columns ---
# Adjust this list depending on what’s irrelevant for modeling
drop_cols = ["user_id", "offline_listening"]  # example from your screenshot
df_model = df_raw.drop(columns=[c for c in drop_cols if c in df_raw.columns])

# --- 3. Separate features and target ---
target_col = "is_churned"
if target_col not in df_model.columns:
    raise KeyError(f"❌ Target column '{target_col}' not found. Available columns: {df_model.columns.tolist()}")

X = df_model.drop(columns=[target_col])
y = df_model[target_col]

# --- 4. One-Hot Encoding for categorical columns ---
categorical_cols = X.select_dtypes(include=["object", "category"]).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print(f"✅ One-hot encoded {len(categorical_cols)} categorical columns.")

# --- 5. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 6. Standard Scaling for numerical columns ---
quantitative_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_train[quantitative_cols] = scaler.fit_transform(X_train[quantitative_cols])
X_test[quantitative_cols] = scaler.transform(X_test[quantitative_cols])
print("✅ Features scaled successfully.")

# --- 7. Train Logistic Regression ---
lr_model = LogisticRegression(max_iter=500,class_weight='balanced')
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("\n🔹 Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# --- 8. Train Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\n🌲 Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# --- 9. Save Models ---
os.makedirs("models", exist_ok=True)
joblib.dump(lr_model, "models/model_lr.pkl")
joblib.dump(rf_model, "models/model_rf.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("✅ Models and scaler saved to training-api/models/")
