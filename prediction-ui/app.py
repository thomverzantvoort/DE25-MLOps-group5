# prediction-ui/app.py
import json
import os
import logging
import requests
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

@app.route('/checkchurn', methods=["GET", "POST"])
def check_churn():
    if request.method == "GET":
        return render_template("input_form_page.html")

    elif request.method == "POST":
        # ✅ Only include features your model was trained on
        prediction_input = [
            {
                "age": int(request.form.get("age")),
                "listening_time": float(request.form.get("listening_time")),
                "songs_played_per_day": float(request.form.get("songs_played_per_day")),
                "skip_rate": float(request.form.get("skip_rate")),
                "ads_listened_per_week": float(request.form.get("ads_listened_per_week"))
            }
        ]


        app.logger.debug("Prediction input: %s", prediction_input)

        # ✅ Get API URL from environment variable (set in Docker)
        predictor_api_url = os.environ['PREDICTOR_API']
        app.logger.debug("Sending request to Predictor API: %s", predictor_api_url)

        # ✅ Send request to prediction API
        res = requests.post(predictor_api_url, json=json.loads(json.dumps(prediction_input)))

        if res.status_code != 200:
            app.logger.error("Prediction API returned error: %s", res.text)
            return jsonify(error="Prediction API error", details=res.text), res.status_code

        # ✅ Parse predictions from API response
        predictions = res.json()
        app.logger.debug("Prediction response: %s", predictions)

        lr_pred = predictions["logistic_regression_prediction"][0]
        rf_pred = predictions["random_forest_prediction"][0]

        # ✅ Combine predictions logically
        churn_status = True if (lr_pred == 1 or rf_pred == 1) else False

        return render_template("response_page.html", churn_variable=churn_status)

    else:
        return jsonify(message="Method Not Allowed"), 405


if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
