import argparse

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.feature_extractor import FeatureExtractor
from src.utils import load_json, load_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--library", default="library.json", help="Path to library.")
    parser.add_argument("-q", "--query", required=True, help="Query image.")
    parser.add_argument("-s", "--min_similarity", default=0.5, type=float, help="Similarity threshold for query.")
    parser.add_argument("-n", "--n_samples", default=10, type=int,
                        help="Max number of samples returned as search result")
    args = parser.parse_args()

    query_image = load_image(args.query)
    library = load_json(args.library)
    paths = library["paths"]
    library_features = np.array(library["features"])

    extractor = FeatureExtractor()
    feature_vector = np.array(extractor.extract(query_image)).reshape(1, -1)
    similarities = cosine_similarity(feature_vector, library_features).reshape(-1)
    indices = np.flip(np.argsort(similarities))
    similarities_sorted = similarities[indices]
    indices = indices[similarities_sorted > args.min_similarity]

    query_resized = cv2.resize(np.array(query_image), (512, 512))
    cv2.imshow("Query", query_resized)

    for i in indices[:args.n_samples]:
        score = similarities[i]
        path = paths[i]
        image = load_image(path)
        image_resized = cv2.resize(np.array(image), (512, 512))
        cv2.imshow(f"Score: {score}", image_resized)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
