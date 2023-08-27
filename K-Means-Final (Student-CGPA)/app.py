from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def k_means(data, k, initial_centers, test_point):
    centers = np.array(initial_centers)
    prev_centers = np.zeros(centers.shape)
    clusters = {i: [] for i in range(k)}

    while not np.all(np.isclose(centers, prev_centers)):
        prev_centers = centers.copy()
        clusters = {i: [] for i in range(k)}

        for point in data:
            distances = [euclidean_distance(point, center) for center in centers]
            closest_center = np.argmin(distances)
            clusters[closest_center].append(point)

        centers = [np.mean(cluster, axis=0) if cluster else centers[i] for i, cluster in clusters.items()]

    distances_to_centers = [euclidean_distance(test_point, center) for center in centers]
    closest_cluster = np.argmin(distances_to_centers)

    return clusters, closest_cluster

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_clusters = int(request.form['num_clusters'])
        initial_centers = [float(center) for center in request.form['centers'].split(',')]
        test_point = np.array([float(coord) for coord in request.form['test_point'].split(',')])

        data = np.array([
            [3.45], [3.78], [2.98], [3.24], [4.0], [3.9]
        ])  # CGPA data points

        clusters, closest_cluster = k_means(data, num_clusters, initial_centers, test_point)

        return render_template('clusters.html', clusters=clusters, closest_cluster=closest_cluster, input_given=True)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
