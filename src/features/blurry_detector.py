import cv2

from src.utils import pil_to_array


class BlurryDetector:

    def run(self, image):
        image_array = pil_to_array(image)
        return cv2.Laplacian(image_array, cv2.CV_64F).var()
