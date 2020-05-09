from src.face_detection.face_detector import FaceDetector
from src.features.blurry_detector import BlurryDetector
from src.features.feature_extractor import FeatureExtractor
from src.object_detection.object_detector import ObjectDetector
from src.tagging.tagger import Tagger


class ImageProcessor:

    def __init__(self):
        self.pipeline = {
            "feature_vector": FeatureExtractor(),
            "object_detections": ObjectDetector(),
            "face_detections": FaceDetector(),
            "tags": Tagger(device="cpu"),
            "blurriness": BlurryDetector(),
        }


    def process(self, image):
        results = {}
        for key, processor in self.pipeline.items():
            results[key] = processor.run(image)
        results["image_size"] = list(image.size)
        return results
