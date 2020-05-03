import os

import numpy as np
import cv2

from src.utils import pil_to_array


class FaceDetector:

    def __init__(self, confidence_threshold=0.8):
        base_path = os.path.split(os.path.abspath(__file__))[0]
        prototxt_path = os.path.join(base_path, "deploy.prototxt.txt")
        model_path = os.path.join(base_path, "res10_300x300_ssd_iter_140000.caffemodel")
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.confidence_threshold = confidence_threshold

    def run(self, image):
        image = pil_to_array(image)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                bbox = box.astype("int")

                results.append(
                    {
                        "confidence": confidence,
                        "bbox": bbox.tolist(),
                    }
                )

        return results
