
from flask import Flask, request, render_template
import numpy as np
import joblib

model = joblib.load("/workspaces/ML-web-app-using-Flask/models/decision_tree_model.pkl")


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            gender = float(request.form["gender"])
            impluse = float(request.form["impluse"])
            pressurehight = float(request.form["pressurehight"])
            pressurelow = float(request.form["pressurelow"])
            glucose = float(request.form["glucose"])
            kcm = float(request.form["kcm"])
            troponin = float(request.form["troponin"])

            
            input_data = np.array([[age, gender, impluse, pressurehight, pressurelow, glucose, kcm, troponin]])

            prediction = model.predict(input_data)[0]
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(debug=True)

