import numpy as np
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def KNeighborsClassifier(X_train, y_train, x_test, k):
    distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = np.bincount(k_nearest_labels).argmax()
    return most_common
