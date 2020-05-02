import sys

sys.path.append('./src/object_detection/PyTorchYOLOv3')
sys.path.append('./src/object_detection')

from constants import EXTENSIONS
from src.image_processor import ImageProcessor
from src.utils import recursive_glob, load_image, collect_file_info

processor = ImageProcessor()
paths = recursive_glob("dataset", EXTENSIONS)

for i, path in enumerate(paths):
    print(f"{i + 1}/{len(paths)}")
    image = load_image(path)
    results = processor.process(image)
    file_info = collect_file_info(path)
    print(results["size"])
