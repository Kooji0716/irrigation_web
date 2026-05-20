from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html")


@app.route("/sklearn-models")
def sklearn_models():
    return render_template("sklearn_models.html")


@app.route("/neural-network")
def neural_network():
    return render_template("neural_network.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        model_type = request.form.get("model_type")

        if model_type == "rf":
            result = "Medium"
            explanation = "目前使用的是非神經網路模型示範結果。正式版本會串接 Random Forest New 模型。"
        else:
            result = "Medium"
            explanation = "目前使用的是神經網路模型示範結果。正式版本會串接 Keras / TensorFlow 模型。"

        return render_template(
            "predict.html",
            result=result,
            explanation=explanation
        )

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)