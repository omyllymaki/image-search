import argparse
import os
import sys

import cv2

from src.object_detection.plotting import add_detection_boxes_to_image

sys.path.append('./src/object_detection/PyTorchYOLOv3')
sys.path.append('./src/object_detection')

from src.object_detection.object_detector import ObjectDetector
from src.utils import get_file_paths, load_image, pil_to_array
import numpy as np

EXTENSIONS = (".jpg", ".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset.")
    ap.add_argument("-t", "--threshold", default=0.8, type=float, help="Threshold value for detections.")
    ap.add_argument('--show', dest='show', action='store_true', help="Option for showing detection", default=False)
    args = ap.parse_args()

    paths = get_file_paths(args.dataset, EXTENSIONS)
    detector = ObjectDetector(confidence_threshold=args.threshold, model_image_size=608)
    for i, p in enumerate(paths):
        print(f"{i + 1}/{len(paths)}")
        image = load_image(p)
        if image is not None:
            detections = detector.run(image)
            if len(detections) > 0:
                detected_classes = list(set([d["class"] for d in detections]))
                _, filename = os.path.split(p)
                print(f"{filename}: {detected_classes}")
                if args.show:
                    image_array = pil_to_array(image)
                    image_array = add_detection_boxes_to_image(image_array, detections)
                    image_array = cv2.resize(image_array, (500, 500))
                    cv2.imshow(f"{filename}", image_array)
                    cv2.waitKey(0)


if __name__ == "__main__":
    main()
