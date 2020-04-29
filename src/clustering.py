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

    distance = calculate_distance(linkage_matrix)
    clusters = fcluster(linkage_matrix, distance, criterion='distance')
    return clusters


def calculate_distance(linkage_matrix, threshold=0.2):
    distances = linkage_matrix[:, 2]
    distances_normed = distances / distances.max()
    return distances[distances_normed > threshold][0]


def calculate_linkage_matrix(X):
    similarities = cosine_similarity(X)
    distances = 1 - similarities
    linkage_matrix = hc.linkage(distances, 'ward')
    return linkage_matrix
