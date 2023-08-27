import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def KNeighborsClassifier(training_data, new_point, k):
    distances = [euclidean_distance(new_point, data_point[:2]) for data_point in training_data]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [training_data[i][2] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common