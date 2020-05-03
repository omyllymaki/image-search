import os
import sys

import torch
from torchvision import transforms

from src.object_detection.PyTorchYOLOv3.models import Darknet
from src.object_detection.PyTorchYOLOv3.utils.utils import load_classes, non_max_suppression
from src.object_detection.constants import CLASSES_PATH, CONFIG_PATH, WEIGHTS_PATH, MODEL_IMAGE_SIZE, DEVICE, \
    CONFIDENCE_THRESHOLD, SUPRESSION_THRESHOLD, CLASS_SCORE_THRESHOLD
import numpy as np

sys.path.append('./PyTorchYOLOv3')


class ObjectDetector:

    def __init__(self,
                 confidence_threshold=CONFIDENCE_THRESHOLD,
                 supression_threshold=SUPRESSION_THRESHOLD,
                 class_score_threshold=CLASS_SCORE_THRESHOLD,
                 model_image_size=MODEL_IMAGE_SIZE):
        self.confidence_threshold = confidence_threshold
        self.supression_threshold = supression_threshold
        self.class_score_threshold = class_score_threshold
        self.model_image_size = model_image_size
        base_path = os.path.split(os.path.abspath(__file__))[0]
        classes_path = os.path.join(base_path, CLASSES_PATH)
        config_path = os.path.join(base_path, CONFIG_PATH)
        weights_path = os.path.join(base_path, WEIGHTS_PATH)
        self.classes = load_classes(classes_path)
        self.model = self._load_pretrained_darknet_model(config_path, weights_path, self.model_image_size)
        self.model.to(DEVICE)

    def run(self, image):
        image_tensor = self._process_image(image, self.model_image_size).to(DEVICE)
        detections = self._detect_image_objects(image_tensor)
        results = []
        if detections is not None:
            for detection in detections:
                class_score = detection[5].item()
                if class_score > self.class_score_threshold:
                    result = {"bbox": self._get_bbox(np.array(image).shape, detection[:4]),
                              "class": self.classes[int(detection[6])],
                              "object_confidence": detection[4].item(),
                              "class_score": class_score}
                    results.append(result)
        return results

    def _detect_image_objects(self, image):
        with torch.no_grad():
            detections = self.model(image)
            detections = non_max_suppression(detections, self.confidence_threshold, self.supression_threshold)
        return detections[0]

    def _load_pretrained_darknet_model(self, config_path, weights_path, image_size):
        model = Darknet(config_path, img_size=image_size)
        model.load_darknet_weights(weights_path)
        model.eval()
        return model

    @staticmethod
    def _process_image(image, image_size):
        ratio = min(image_size / image.size[0], image_size / image.size[1])
        imw = round(image.size[0] * ratio)
        imh = round(image.size[1] * ratio)

        img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                             transforms.Pad((max(int((imh - imw) / 2), 0),
                                                             max(int((imw - imh) / 2), 0),
                                                             max(int((imh - imw) / 2), 0),
                                                             max(int((imw - imh) / 2), 0)),
                                                            (128, 128, 128)),
                                             transforms.ToTensor(),
                                             ])

        image_tensor = img_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        return image_tensor

    def _get_bbox(self, image_size, detection_coordinates):
        x1, y1, x2, y2 = detection_coordinates

        pad_x = max(image_size[0] - image_size[1], 0) * (self.model_image_size / max(image_size))
        pad_y = max(image_size[1] - image_size[0], 0) * (self.model_image_size / max(image_size))
        unpad_h = self.model_image_size - pad_y
        unpad_w = self.model_image_size - pad_x

        box_h = ((y2 - y1) / unpad_h) * image_size[0]
        box_w = ((x2 - x1) / unpad_w) * image_size[1]
        y1 = ((y1 - pad_y // 2) / unpad_h) * image_size[0]
        x1 = ((x1 - pad_x // 2) / unpad_w) * image_size[1]

        return [int(x1), int(y1), int(box_w + x1), int(box_h + y1)]
