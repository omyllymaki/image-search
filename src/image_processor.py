from src.features.blurry_detector import BlurryDetector
from src.features.feature_extractor import FeatureExtractor
from src.object_detection.object_detector import ObjectDetector


class ImageProcessor:

    def __init__(self):
        self.pipeline = {
            "features": FeatureExtractor(),
            "detections": ObjectDetector(),
            "blurriness": BlurryDetector(),
        }

    def process(self, image):
        results = {}
        for key, processor in self.pipeline.items():
            results[key] = processor.run(image)
        results["size"] = list(image.size)
        return results
