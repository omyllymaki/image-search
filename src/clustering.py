import numpy as np
from scipy.cluster import hierarchy as hc
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics.pairwise import cosine_similarity


def calculate_clusters(linkage_matrix, max_distance=None, n_clusters=None):
    if max_distance:
        clusters = fcluster(linkage_matrix, max_distance, criterion='distance')
        return clusters

    if n_clusters:
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        return clusters

    n_clusters = calculate_optimal_number_of_clusters(linkage_matrix)
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    return clusters


def calculate_optimal_number_of_clusters(linkage_matrix):
    distances = linkage_matrix[:, 2]
    acceleration = np.diff(distances, 2)
    acceleration_reversed = acceleration[::-1]
    n_clusters = acceleration_reversed.argmax() + 2
    return n_clusters


def calculate_linkage_matrix(X):
    similarities = cosine_similarity(X)
    distances = 1 - similarities
    linkage_matrix = hc.linkage(distances, 'ward')
    return linkage_matrix
