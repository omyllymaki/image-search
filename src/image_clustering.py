from src.clustering import kmeans, hierarchical
from src.clustering.hierarchical import calculate_linkage_matrix
from src.clustering.visualization import plot_linkage_data
import matplotlib.pyplot as plt


class ImageClustering:

    def __init__(self, feature_array):
        self.feature_array = feature_array

    def kmeans_clustering(self, k_candidates=range(3, 31)):
        clusters = kmeans.calculate_clusters(self.feature_array, k_candidates)
        return clusters

    def hierarchical_clustering(self, distance=None):
        linkage_matrix = calculate_linkage_matrix(self.feature_array)
        clusters = hierarchical.calculate_clusters(linkage_matrix, distance)
        return clusters

    def show_dendogram(self):
        linkage_matrix = calculate_linkage_matrix(self.feature_array)
        plot_linkage_data(linkage_matrix)
        plt.show()
