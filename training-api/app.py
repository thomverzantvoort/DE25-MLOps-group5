# training-api/app.py
import os
from flask import Flask, jsonify
from train import main as train_main  # We'll define 'main()' inside train.py

app = Flask(__name__)

@app.route('/train', methods=['GET'])
def train_model():
    """Trigger model training when this endpoint is called."""
    try:
        metrics = train_main()  # run training from train.py
        return jsonify({"message": "✅ Model training completed successfully", "metrics": metrics}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
