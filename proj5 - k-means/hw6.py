import numpy as np
from skimage import io


def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image.
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids.
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array.
    '''

    centroids = []
    # get k random pixels for centroids
    random_points = np.random.randint(0, X.shape[0], k)
    centroids = X[random_points, :]
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float64)


def lp_distance(X, centroids, p=2):
    '''
    Inputs:
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)' thats holds the distances of
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    # Add new axes to enable the calculation
    X = X[np.newaxis]
    centroids1 = centroids[:, np.newaxis]
    # Formula from class
    differences = np.abs(X - centroids1)
    distances = np.sum(differences ** p, axis=2) ** (1 / p)
    return distances


def kmeans(X, k, p, max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    for i in range(max_iter):
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)  # we get to all pixels and their closest centroid
        # we calc from the data the average of each pixels class according to the  classes which is the new centroids
        old_centroids = centroids
        centroids = np.array([np.mean(X[classes == j, :], axis=0) for j in range(k)])

        if np.all(centroids == old_centroids):  # no change in the centroids
            break

    return centroids, classes


def kmeans_pp(X, k, p=2, max_iter=100):
    """
    Implementation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """

    centroids2 = []
    chosen_indices = set()
    X_copy = X  # copy for deleting the chosen centroids
    # Choose the first centroid randomly
    first_centroid_idx = np.random.randint(0, len(X))
    centroids2.append(X[first_centroid_idx])
    X_copy = np.delete(X_copy, first_centroid_idx, axis=0)  # remove the chosen centroid for the procces

    for _ in range(1, k):
        # Calculate the distance of each point to the nearest chosen centroid

        distances = lp_distance(X_copy, np.array(centroids2), p)
        min_distances = np.min(distances, axis=0)
        squared_distances = min_distances ** 2

        # Calculate the probabilities according to the squared distances
        sum_squared_distances = np.sum(squared_distances)
        probabilities = squared_distances / sum_squared_distances

        # Select the next centroid with the calculated probabilities
        next_centroid_idx = np.random.choice(len(X_copy), p=probabilities)
        # Ensure the next centroid is not already chosen
        centroids2.append(X_copy[next_centroid_idx])
        X_copy = np.delete(X_copy, next_centroid_idx, axis=0)

    centroids2 = np.array(centroids2)

    # Proceed with the standard k-means clustering
    return kmeans_for_pp(X, centroids2, k, p, max_iter)


# standard kmeans function with starting centroids
def kmeans_for_pp(X, centroids, k, p, max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = centroids
    for i in range(max_iter):
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)  # we get to all pixels and their closest centroid
        # we calc from the data the average of each pixels class according to the  classes which is the new centroids
        old_centroids = centroids
        centroids = np.array([np.mean(X[classes == j, :], axis=0) for j in range(k)])

        if np.all(centroids == old_centroids):
            break

    return centroids, classes




