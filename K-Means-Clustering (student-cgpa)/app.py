from flask import Flask, render_template, request

app = Flask(__name__)

def euclidean_distance(point1, point2):
    return abs(point1 - point2)

def assign_to_clusters(data, centers):
    clusters = {}
    for point in data:
        distances = [euclidean_distance(point, center) for center in centers]
        cluster_idx = distances.index(min(distances))
        if cluster_idx not in clusters:
            clusters[cluster_idx] = []
        clusters[cluster_idx].append(point)
    return clusters

def update_centers(clusters):
    centers = []
    for cluster_points in clusters.values():
        center = sum(cluster_points) / len(cluster_points)
        centers.append(center)
    return centers

def k_means(data, initial_centers, num_clusters, max_iterations=100):
    centers = initial_centers
    for _ in range(max_iterations):
        clusters = assign_to_clusters(data, centers)
        new_centers = update_centers(clusters)
        if new_centers == centers:
            break
        centers = new_centers
    return clusters, centers

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_clusters = int(request.form['num_clusters'])
        initial_centers = [float(center) for center in request.form['centers'].split(',')]
        data = [3.45, 3.78, 2.98, 3.24, 4.0, 3.9]
        clusters, final_centers = k_means(data, initial_centers, num_clusters)
        return render_template('clusters.html', clusters=clusters, centers=final_centers, num_clusters=num_clusters, input_given=True)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
