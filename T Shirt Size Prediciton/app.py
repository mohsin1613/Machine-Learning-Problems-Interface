import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, request, render_template

X_train = np.array([[158, 58], [158, 59], [158, 63], [160, 59], [160, 60], [163, 60], [163, 61], [160, 64],
                    [163, 64], [165, 61], [165, 62], [165, 65], [168, 62], [168, 63], [168, 66], [170, 63],
                    [170, 64], [170, 68]])
y_train = np.array(['M', 'M', 'M', 'M', 'M', 'M', 'M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'])

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    height = int(request.form['height'])
    weight = int(request.form['weight'])
    k = int(request.form['clusters'])
    
    if k % 2 == 0:
        message = "Please Enter Odd Number!"
        return render_template('index.html', message=message)
    
    new_data = np.array([[height, weight]])
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    prediction = knn.predict(new_data)
    
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
