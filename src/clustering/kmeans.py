from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def calculate_clusters(features, n_cluster_candidates=None, metric="cosine"):
    if n_cluster_candidates is None:
        n_cluster_candidates = range(3, features.shape[0])

    max_score = 0
    result = None
    for n_clusters in n_cluster_candidates:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(features)
        labels = kmeans.labels_
        score = silhouette_score(features, labels, metric=metric)
        if score > max_score:
            result = labels
            max_score = score

    return result
