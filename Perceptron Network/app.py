from flask import Flask, request, render_template

app = Flask(__name__)


class Perceptron:
    def __init__(self):
        self.weights = [0.5, 0.5, 0.5]
        self.threshold = 0.0

    def predict(self, inputs):
        activation = sum(w * x for w, x in zip(self.weights, inputs))
        return 1 if activation >= self.threshold else -1


perceptron = Perceptron()


@app.route('/', methods=['GET', 'POST'])
def home():
    fruit_type = None

    if request.method == 'POST':
        shape = int(request.form['shape'])
        texture = int(request.form['texture'])
        weight = int(request.form['weight'])

        result = perceptron.predict([shape, texture, weight])
        fruit_type = "Orange" if result == 1 else "Apple"

    return render_template('index.html', fruit_type=fruit_type)


if __name__ == '__main__':
    app.run(debug=True)
