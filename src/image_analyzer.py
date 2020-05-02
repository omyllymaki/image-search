import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.clustering import kmeans, hierarchical
from src.clustering.hierarchical import calculate_linkage_matrix
from src.clustering.visualization import plot_linkage_data
import matplotlib.pyplot as plt

from src.database.db_reader import DBReader
from src.features.feature_extractor import FeatureExtractor


class DBAnalyzer:

    def __init__(self, engine):
        self.feature_extractor = FeatureExtractor()
        self.db_reader = DBReader(engine)

        data = self.db_reader.get_feature_vectors()
        self.feature_array = [d["feature_vector"] for d in data]
        self.paths = [d["path"] for d in data]

    def kmeans_clustering(self, k_candidates=range(3, 31)):
        clusters = kmeans.calculate_clusters(self.feature_array, k_candidates)
        return self._cluster_paths(clusters)

    def hierarchical_clustering(self, distance=None):
        linkage_matrix = calculate_linkage_matrix(self.feature_array)
        if distance is None:
            plot_linkage_data(linkage_matrix)
            plt.show()
            distance = float(input('Enter threshold distance for clustering:'))
        clusters = hierarchical.calculate_clusters(linkage_matrix, distance)
        return self._cluster_paths(clusters)

    def image_query(self, query_image, min_similarity=0.50, max_samples=10):
        feature_vector = np.array(self.feature_extractor.run(query_image)).reshape(1, -1)
        similarities = cosine_similarity(feature_vector, self.feature_array).reshape(-1)
        indices = np.flip(np.argsort(similarities))
        similarities_sorted = similarities[indices]
        indices = indices[similarities_sorted > min_similarity][:max_samples]
        query_paths = [self.paths[i] for i in indices]
        query_scores = [similarities[i] for i in indices]
        return query_paths, query_scores

    def _cluster_paths(self, clusters):
        clustered_paths = []
        for c in set(clusters):
            clustered_paths.append(np.array(self.paths)[clusters == c].tolist())
        return clustered_paths

