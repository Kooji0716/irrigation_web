from flask import Flask, render_template

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


if __name__ == "__main__":
    app.run(debug=True)