import pandas as pd
import joblib

from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load trained machine learning model
model = joblib.load("model.pkl")

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Churn Prediction</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 0;
        }

        .header {
            background: #1f2937;
            color: white;
            padding: 25px;
            text-align: center;
        }

        .container {
            width: 90%;
            max-width: 1100px;
            margin: 30px auto;
        }

        .cards {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }

        .card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }

        .card h2 {
            margin: 5px;
            color: #2563eb;
        }

        .section {
            background: white;
            margin-top: 25px;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }

        label {
            display: block;
            margin-top: 15px;
            font-weight: bold;
        }

        input, select {
            width: 100%;
            padding: 10px;
            margin-top: 6px;
            border: 1px solid #ccc;
            border-radius: 6px;
            box-sizing: border-box;
        }

        button {
            width: 100%;
            margin-top: 25px;
            padding: 12px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        .result {
            margin-top: 25px;
            padding: 20px;
            background: #eef2ff;
            border-radius: 10px;
            text-align: center;
        }

        @media(max-width: 800px) {
            .cards {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>

<body>

<div class="header">
    <h1>Customer Churn Prediction </h1>
    <p>Machine Learning Model: Logistic Regression</p>
</div>

<div class="container">

    <div class="cards">

        <div class="card">
            <p>Total Customers</p>
            <h2>{{ total_customers }}</h2>
        </div>

        <div class="card">
            <p>Churned Customers</p>
            <h2>{{ churned_customers }}</h2>
        </div>

        <div class="card">
            <p>Retained Customers</p>
            <h2>{{ retained_customers }}</h2>
        </div>

        <div class="card">
            <p>Churn Rate</p>
            <h2>{{ churn_rate }}%</h2>
        </div>

    </div>


    <div class="section">

        <h2>Predict Customer Churn</h2>

        <form method="POST">

            <label>Tenure (Months)</label>
            <input type="number" name="tenure"
                   min="0" required>

            <label>Monthly Charges</label>
            <input type="number" name="monthly_charges"
                   step="0.01" min="0" required>

            <label>Contract</label>
            <select name="contract" required>
                <option value="Month-to-month">
                    Month-to-month
                </option>
                <option value="One year">
                    One year
                </option>
                <option value="Two year">
                    Two year
                </option>
            </select>

            <label>Internet Service</label>
            <select name="internet_service" required>
                <option value="DSL">DSL</option>
                <option value="Fiber optic">Fiber optic</option>
                <option value="No">No</option>
            </select>

            <label>Payment Method</label>
            <select name="payment_method" required>
                <option value="Electronic check">
                    Electronic check
                </option>
                <option value="Mailed check">
                    Mailed check
                </option>
                <option value="Bank transfer (automatic)">
                    Bank transfer (automatic)
                </option>
                <option value="Credit card (automatic)">
                    Credit card (automatic)
                </option>
            </select>

            <button type="submit">
                Predict Churn
            </button>

        </form>


        {% if prediction is not none %}

        <div class="result">

            <h2>{{ prediction_text }}</h2>

            <h3>
                Churn Probability:
                {{ probability }}%
            </h3>

        </div>

        {% endif %}

    </div>

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    prediction_text = ""
    probability = 0

    if request.method == "POST":

        tenure = float(request.form["tenure"])
        monthly_charges = float(
            request.form["monthly_charges"]
        )

        contract = request.form["contract"]
        internet_service = request.form["internet_service"]
        payment_method = request.form["payment_method"]

        customer = pd.DataFrame({
            "tenure": [tenure],
            "MonthlyCharges": [monthly_charges],
            "Contract": [contract],
            "InternetService": [internet_service],
            "PaymentMethod": [payment_method]
        })

        prediction = model.predict(customer)[0]

        probability = (
            model.predict_proba(customer)[0][1] * 100
        )

        if prediction == 1:
            prediction_text = "⚠️ Customer is likely to CHURN"
        else:
            prediction_text = "✅ Customer is likely to STAY"

        probability = round(probability, 2)

    return render_template_string(
        html,
        total_customers=total_customers,
        churned_customers=churned_customers,
        retained_customers=retained_customers,
        churn_rate=round(churn_rate, 2),
        prediction=prediction,
        prediction_text=prediction_text,
        probability=probability
    )

    if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=10000,
        debug=False
    )
