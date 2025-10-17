# prediction-api/churn_predictor.py
import os
import joblib
import pandas as pd
from flask import jsonify
from io import StringIO
import json

class ChurnPredictor:
    def __init__(self):
        self.lr_model = None
        self.rf_model = None
        self.scaler = None
        self._load_models()

    def _load_models(self):
        """Loads models and scaler from local / mounted directory."""
        model_repo = os.getenv("MODEL_REPO", "models")
        self.lr_model = joblib.load(os.path.join(model_repo, "model_lr.pkl"))
        self.rf_model = joblib.load(os.path.join(model_repo, "model_rf.pkl"))
        self.scaler = joblib.load(os.path.join(model_repo, "scaler.pkl"))
        print(f"✅ Models loaded from {model_repo}")

    def predict_single_record(self, prediction_input):
        """Predict churn for one or more records in JSON"""
        df = pd.read_json(StringIO(json.dumps(prediction_input)), orient='records')

        # Select numeric columns only for scaling
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        # Predict with both models
        pred_lr = self.lr_model.predict(df)
        pred_rf = self.rf_model.predict(df)

        results = {
            "logistic_regression_prediction": pred_lr.tolist(),
            "random_forest_prediction": pred_rf.tolist()
        }

        return results
