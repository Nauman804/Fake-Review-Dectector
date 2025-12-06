from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

model_path = os.path.join("..", "model", "fake_review_model.pkl")

with open(model_path, "rb") as f:
    vectorizer, model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]
    if review.strip() == "":
        prediction = "Please enter a review!"
    else:
        transformed = vectorizer.transform([review])
        result = model.predict(transformed)[0]
        prediction = "❌ Fake Review" if result == 1 else "✔️ Genuine Review"
    return render_template("index.html", prediction=prediction, review_text=review)

if __name__ == "__main__":
    app.run(debug=True)
