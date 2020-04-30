import argparse
import os

import cv2

from src.utils import get_file_paths

EXTENSIONS = (".jpg", ".png")


def calculate_blurriness_score(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset.")
    ap.add_argument("-t", "--threshold", required=True, type=float, help="Threshold value for blurriness.")
    args = ap.parse_args()

    paths = get_file_paths(args.dataset, EXTENSIONS)
    blurry_images, blurry_image_paths, blurry_image_scores = [], [], []
    for i, p in enumerate(paths):
        print(f"{i + 1}/{len(paths)}")
        image = cv2.imread(p)
        if image is not None:
            score = calculate_blurriness_score(image)
            if score < args.threshold:
                blurry_images.append(image)
                blurry_image_paths.append(p)
                blurry_image_scores.append(score)

    for p, image, score in zip(blurry_image_paths, blurry_images, blurry_image_scores):
        image = cv2.resize(image, (500, 500))
        _, filename = os.path.split(p)
        cv2.imshow(f"{filename}; score {score:0.1f}", image)
        response = cv2.waitKey(0)
        if response == 100:  # d key
            print(f"Removing file {p}")
            os.remove(p)


if __name__ == "__main__":
    main()
