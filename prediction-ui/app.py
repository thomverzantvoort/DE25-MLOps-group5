# prediction-ui/app.py
import json
import os
import logging
import requests
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route('/checkchurn', methods=["GET", "POST"])
def check_churn():
    if request.method == "GET":
        return render_template("input_form_page.html")

    elif request.method == "POST":
        # Build input data from form
        prediction_input = [
            {
                "age": int(request.form.get("age")),
                "subscription_length": int(request.form.get("subscription_length")),
                "plays_per_day": int(request.form.get("plays_per_day")),
                "gender_Male": int(request.form.get("gender_Male"))
            }
        ]

        app.logger.debug("Prediction input : %s", prediction_input)

        # Get API URL from environment variable (set in Docker)
        predictor_api_url = os.environ['PREDICTOR_API']
        res = requests.post(predictor_api_url, json=json.loads(json.dumps(prediction_input)))

        # Read predictions from API response
        predictions = res.json()
        lr_pred = predictions["logistic_regression_prediction"][0]
        rf_pred = predictions["random_forest_prediction"][0]

        # For simplicity, combine both
        churn_status = True if (lr_pred == 1 or rf_pred == 1) else False

        return render_template("response_page.html", churn_variable=churn_status)

    else:
        return jsonify(message="Method Not Allowed"), 405


if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
