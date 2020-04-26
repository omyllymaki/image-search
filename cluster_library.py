import argparse
import os

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
    parser.add_argument("-o", "--output", default="clustering_output", help="Path for clustering results.")
    parser.add_argument('--select_distance', dest='select_distance', action='store_true',
                        help="Option for selecting threshold distance based on dendogram figure")
    parser.add_argument('--show_examples', dest='show_examples', action='store_true',
                        help="Option for showing examples")
    parser.add_argument("--save", dest='save', action='store_true', help="Option for saving results")
    parser.set_defaults(select_distance=False, save=False, show_examples=False)
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

    # Just show some examples
    for c in unique_clusters:
        cluster_paths = np.array(paths)[clusters == c]
        image = np.zeros((150, 150, 3), np.uint8)
        montage = cv2.resize(image, (150, 150))
        for p in cluster_paths[1:args.n_samples + 1]:
            image = cv2.imread(p)
            image = cv2.resize(image, (150, 150))
            montage = np.hstack([montage, image])
        cv2.imshow(f"Cluster {c}", montage)
        cv2.waitKey(0)

    # Create new images based on clustering results
    if args.save:
        print(f"Saving results to {args.output}")
        os.makedirs(args.output, exist_ok=True)
        for c in unique_clusters:
            print(f"Saving results for cluster {c}")
            cluster_folder_path = os.path.join(args.output, "cluster_" + str(c))
            os.makedirs(cluster_folder_path, exist_ok=True)
            paths_for_cluster = np.array(paths)[clusters == c]
            for p in paths_for_cluster:
                image = cv2.imread(p)
                base, filename = os.path.split(p)
                new_path = os.path.join(cluster_folder_path, filename)
                cv2.imwrite(new_path, image)


if __name__ == "__main__":
    main()
