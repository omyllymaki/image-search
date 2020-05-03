import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class DuplicateDetector:

    def __init__(self, feature_array):
        self.feature_array = feature_array

    def find_duplicates(self, similarity_threshold=0.99):
        similarities = cosine_similarity(self.feature_array)
        np.fill_diagonal(similarities, 0)

        duplicates = {}
        for k, row in enumerate(similarities):
            duplicates[k] = np.where(row > similarity_threshold)[0]

        return duplicates
