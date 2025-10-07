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
df = pd.read_csv(DATA_PATH)

print("📂 Current working directory:", os.getcwd())
print("📄 Reading file from:", DATA_PATH)
print("🧾 Columns found:", df.columns.tolist())
print("Data shape:", df.shape)


# --- 2. Basic Cleaning ---
df = df.dropna()
# adjust this to your real target column name
target_col = "is_churned"
X = df.drop(columns=[target_col])
y = df[target_col]

# --- 3. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Train Logistic Regression ---
lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# --- 6. Train Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# --- 7. Save Models ---
os.makedirs("models", exist_ok=True)
joblib.dump(lr_model, "models/model_lr.pkl")
joblib.dump(rf_model, "models/model_rf.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Models and scaler saved to training-api/models/")


