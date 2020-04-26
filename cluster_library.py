import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.clustering import calculate_clusters, calculate_linkage_matrix
from src.utils import load_json
from src.visualization import plot_linkage_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--library", default="library.json", help="Path to library.")
    parser.add_argument("-n", "--n_samples", default=10, type=int, help="Number of samples to show for every cluster.")
    parser.add_argument("-d", "--distance", default=-1, type=int, help="Distance threshold for clustering.")
    parser.add_argument('--select_distance', dest='select_distance', action='store_true')
    parser.set_defaults(select_distance=False)
    args = parser.parse_args()

    library = load_json(args.library)
    paths = library["paths"]
    features = np.array(library["features"])

    linkage_matrix = calculate_linkage_matrix(features)

    threshold_distance = None
    if args.distance > 0:
        threshold_distance = args.distance
    if args.select_distance:
        plot_linkage_data(linkage_matrix)
        plt.show()
        threshold_distance = float(input('Enter threshold distance for clustering:'))

    clusters = calculate_clusters(linkage_matrix, threshold_distance)

    unique_clusters = set(clusters)
    print(f"Found {len(unique_clusters)} clusters")
    for c in unique_clusters:
        cluster_paths = np.array(paths)[clusters == c]
        image = cv2.imread(cluster_paths[0])
        montage = cv2.resize(image, (150, 150))
        for p in cluster_paths[1:args.n_samples + 1]:
            image = cv2.imread(p)
            image = cv2.resize(image, (150, 150))
            montage = np.hstack([montage, image])
        cv2.imshow(f"Cluster {c}", montage)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
