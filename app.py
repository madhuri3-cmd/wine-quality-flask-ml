from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained ML model
model = joblib.load("wine_quality_model.joblib")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    fixed_acidity = float(request.form["fixed_acidity"])
    volatile_acidity = float(request.form["volatile_acidity"])
    citric_acid = float(request.form["citric_acid"])
    residual_sugar = float(request.form["residual_sugar"])
    chlorides = float(request.form["chlorides"])
    free_sulfur_dioxide = float(request.form["free_sulfur_dioxide"])
    total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
    density = float(request.form["density"])
    ph = float(request.form["ph"])
    sulphates = float(request.form["sulphates"])
    alcohol = float(request.form["alcohol"])

    features = [[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        ph,
        sulphates,
        alcohol
    ]]

    prediction = model.predict(features)[0]

    return render_template(
        "result.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)