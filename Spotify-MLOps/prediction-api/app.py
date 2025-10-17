# prediction-api/app.py
import os
from flask import Flask, request, jsonify
from churn_predictor import ChurnPredictor

app = Flask(__name__)
app.config["DEBUG"] = True

# initialize predictor (loads models lazily)
cp = ChurnPredictor()

@app.route('/predict/', methods=['POST'])
def predict_single():
    """Accept JSON input and return churn prediction"""
    try:
        input_json = request.get_json()
        result = cp.predict_single_record(input_json)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=int(os.getenv("PORT", 5000)), host='0.0.0.0', debug=True)
