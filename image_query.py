import argparse
import os

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.feature_extractor import FeatureExtractor
from src.utils import load_json, load_image
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--library", default="library.json", help="Path to library.")
    parser.add_argument("-q", "--query", required=True, help="Query image.")
    parser.add_argument("-s", "--min_similarity", default=0.5, type=float, help="Similarity threshold for query.")
    parser.add_argument("-n", "--n_samples", default=10, type=int,
                        help="Max number of samples returned as search result")
    parser.add_argument("-o", "--output", default="query_output", help="Path for query results.")
    parser.add_argument("--save", dest='save', action='store_true', help="Option for saving results")
    parser.set_defaults(save=False)
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

    if args.save:
        print(f"Saving results to {args.output}")
        os.makedirs(args.output, exist_ok=True)

    query_image = cv2.cvtColor(np.array(query_image), cv2.COLOR_RGB2BGR)
    query_resized = cv2.resize(query_image, (512, 512))
    cv2.imshow("Query", query_resized)

    for i in indices[:args.n_samples]:
        score = similarities[i]
        path = paths[i]
        image = cv2.imread(path)
        image_resized = cv2.resize(np.array(image), (512, 512))
        cv2.imshow(f"Score: {score}", image_resized)
        cv2.waitKey(0)

        if args.save:
            base, filename = os.path.split(path)
            new_path = os.path.join(args.output, filename)
            cv2.imwrite(new_path, np.array(image))


if __name__ == "__main__":
    main()
