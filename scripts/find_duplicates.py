import argparse
import os

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import load_json, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--library", default="library.json", help="Path to library.")
    parser.add_argument("-s", "--similarity_threshold", type=float, default=0.99,
                        help="Similarity to be considered as duplicate.")
    parser.add_argument('--display', dest='display', action='store_true', help="Display duplicates")
    parser.add_argument('--no_display', dest='display', action='store_false', help="Do not display duplicates")
    parser.add_argument('--delete', dest='delete', action='store_true', help="Delete all duplicates")
    parser.set_defaults(delete=False, display=True)
    args = parser.parse_args()

    library = load_json(args.library)
    paths = library["paths"].copy()
    features = np.array(library["features"].copy())

    similarities = cosine_similarity(features)
    np.fill_diagonal(similarities, 0)

    all_duplicates = []
    for k, row in enumerate(similarities):

        if k in all_duplicates:
            continue

        indices = np.where(row > args.similarity_threshold)[0]
        if len(indices) > 0:
            ref_filename = os.path.split(paths[k])[1]
            duplicate_filenames = [os.path.split(paths[i])[1] for i in indices]
            print(f"Photo {ref_filename}; duplicates {duplicate_filenames}")
            all_duplicates = all_duplicates + indices.tolist()
            all_duplicates = list(set(all_duplicates))

            reference_image = cv2.imread(paths[k])
            duplicate_images = cv2.resize(reference_image, (150, 150))

            if args.display:
                for i in indices:
                    p = paths[i]
                    image = cv2.imread(p)
                    image = cv2.resize(image, (150, 150))
                    duplicate_images = np.hstack([duplicate_images, image])

                cv2.imshow(f"Duplicate images for {ref_filename}", duplicate_images)

    if args.display:
        cv2.waitKey(0)

    if args.delete and len(all_duplicates) > 0:
        for i in all_duplicates:
            p = paths[i]
            print(f"Removing file {p}")
            os.remove(p)

        print("Creating new library...")
        new_library = {}
        new_features = library["features"].copy()
        new_paths = library["paths"].copy()
        for index in sorted(all_duplicates, reverse=True):
            del new_features[index]
            del new_paths[index]
        new_library["features"] = new_features
        new_library["paths"] = new_paths
        save_json(args.library, new_library)


if __name__ == "__main__":
    main()
