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
drop_cols = ["user_id", "offline_listening"]
df_model = df_raw.drop(columns=[c for c in drop_cols if c in df_raw.columns])

# --- 3. Keep only the numerical features you want + target ---
selected_features = [
    "age",
    "listening_time",
    "songs_played_per_day",
    "skip_rate",
    "ads_listened_per_week",
]
target_col = "is_churned"

# make sure they exist in your data
missing = [
    col for col in selected_features + [target_col] if col not in df_model.columns
]
if missing:
    raise KeyError(f"❌ Missing columns in dataset: {missing}")

X = df_model[selected_features]
y = df_model[target_col]

# --- 4. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 5. Standard Scaling ---
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_features)
X_test = pd.DataFrame(scaler.transform(X_test), columns=selected_features)
print("✅ Features scaled successfully.")

# --- 6. Train Logistic Regression ---
lr_model = LogisticRegression(max_iter=500, class_weight="balanced")
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("\n🔹 Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# --- 7. Train Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\n🌲 Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# --- 8. Save Models ---
os.makedirs("models", exist_ok=True)
joblib.dump(lr_model, "models/model_lr.pkl")
joblib.dump(rf_model, "models/model_rf.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("✅ Models and scaler saved to training-api/models/")


def main():
    return {
        "logistic_regression_accuracy": round(accuracy_score(y_test, y_pred_lr), 3),
        "random_forest_accuracy": round(accuracy_score(y_test, y_pred_rf), 3),
    }


if __name__ == "__main__":
    main()
