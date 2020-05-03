import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.features.feature_extractor import FeatureExtractor


class ImageSearch:

    def __init__(self, feature_array):
        self.feature_array = feature_array
        self.feature_extractor = FeatureExtractor()

    def query(self, image, min_similarity=0.5, max_samples=10):
        feature_vector = np.array(self.feature_extractor.run(image)).reshape(1, -1)
        similarities = cosine_similarity(feature_vector, self.feature_array).reshape(-1)
        indices = np.flip(np.argsort(similarities))
        similarities_sorted = similarities[indices]
        indices = indices[similarities_sorted > min_similarity][:max_samples]
        return indices, similarities_sorted

