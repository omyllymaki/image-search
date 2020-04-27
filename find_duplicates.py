import argparse
import os

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import load_json


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
    paths = library["paths"]
    features = np.array(library["features"])

    similarities = cosine_similarity(features)
    np.fill_diagonal(similarities, 0)

    all_duplicates = []
    for k, row in enumerate(similarities):

        if k in all_duplicates:
            continue

        indices = np.where(row > args.similarity_threshold)[0]
        if len(indices) > 0:
            print(f"Photo {k}; duplicates {indices}")
            all_duplicates = all_duplicates + indices.tolist()

            reference_image = cv2.imread(paths[k])
            duplicate_images = cv2.resize(reference_image, (150, 150))

            if args.display:
                for i in indices:
                    p = paths[i]
                    image = cv2.imread(p)
                    image = cv2.resize(image, (150, 150))
                    duplicate_images = np.hstack([duplicate_images, image])

                cv2.imshow(f"Duplicate images for {k}", duplicate_images)

    if args.display:
        cv2.waitKey(0)

    if args.delete and len(all_duplicates) > 0:
        for i in all_duplicates:
            p = paths[i]
            print(f"Removing file {p}")
            os.remove(p)


if __name__ == "__main__":
    main()
