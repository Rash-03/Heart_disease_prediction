from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and features
model = joblib.load("models/heart_model.pkl")
features = joblib.load("models/features.pkl")  # list of feature names

features = joblib.load("models/features.pkl")
if 'target' in features:
    features.remove('target')

@app.route("/")
def index():
    return render_template("index.html", features=features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        x_input = [float(data[feature]) for feature in features]
        x_input = np.array(x_input).reshape(1, -1)
        probability = model.predict_proba(x_input)[0][1]
        prediction = int(probability >= 0.5)

        return jsonify({
            "probability": round(probability, 2),
            "risk": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
