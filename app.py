from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# load models and scalers
clf_model = pickle.load(open("notebook/classification_model.pkl", "rb"))
clf_scaler = pickle.load(open("notebook/scaler.pkl", "rb"))

reg_model = pickle.load(open("notebook/regression_model.pkl", "rb"))
reg_scaler = pickle.load(open("notebook/reg_scaler.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    fire_result = None
    fwi_result = None

    if request.method == "POST":
        model_choice = request.form["model"]

        # GENERAL FEATURES ONLY (order MUST match training)
        features = [
            float(request.form["Temperature"]),
            float(request.form["RH"]),
            float(request.form["Ws"]),
            float(request.form["Rain"]),
            float(request.form["FFMC"]),
            float(request.form["DMC"]),
            float(request.form["DC"]),
            float(request.form["ISI"]),
            float(request.form["BUI"])
        ]

        X = np.array(features).reshape(1, -1)

        # classification
        if model_choice in ["classification", "both"]:
            Xc = clf_scaler.transform(X)
            fire_result = clf_model.predict(Xc)[0]

        # regression
        if model_choice in ["regression", "both"]:
            Xr = reg_scaler.transform(X)
            fwi_result = round(float(reg_model.predict(Xr)[0]), 2)

    return render_template(
        "index.html",
        fire_result=fire_result,
        fwi_result=fwi_result
    )

if __name__ == "__main__":
    app.run(debug=True)
