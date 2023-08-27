from flask import Flask, render_template, request
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import cv2
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

app = Flask(__name__)


def load_dataset(folder_path):
    images = []
    labels = []
    class_names = os.listdir(folder_path)

    for class_name in class_names:
        class_path = os.path.join(folder_path, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            # Resize images to a common size
            image = cv2.resize(image, (100, 100))
            images.append(image)
            labels.append(class_name)

    return np.array(images), np.array(labels)


dataset_images, dataset_labels = load_dataset(
    '/home/mohsin/Documents/Lab Project/KNN/PotatoTomato')
dataset_images = dataset_images.reshape(dataset_images.shape[0], -1) / 255.0
train_images, test_images, train_labels, test_labels = train_test_split(
    dataset_images, dataset_labels, test_size=0.2, random_state=42)
knn_model = KNeighborsClassifier()
knn_model.fit(train_images, train_labels)
test_predictions = knn_model.predict(test_images)
accuracy = accuracy_score(test_labels, test_predictions)
print(f"Accuracy on Test Dataset: {accuracy:.2f}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No File!!')

    image = cv2.imdecode(np.fromstring(
        file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    resized_image = cv2.resize(image, (100, 100))
    flattened_image = resized_image.reshape(1, -1) / 255.0
    k_neighbors = request.form.get('k_neighbors',type=int)
    if k_neighbors % 2 == 0:
        error = "Please Enter Valid Number!"
        return render_template('index.html', error=error)
    prediction = knn_model.predict(flattened_image)
    return render_template('index.html', message=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
