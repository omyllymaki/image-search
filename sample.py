import os

import cv2
import numpy as np
from sqlalchemy import create_engine
from src.database.db_reader import DBReader
from src.db_synchronizer import DBSynchronizer
from src.db_analyzer import DBAnalyzer
from src.utils import load_image, pil_to_array


def add_rectangle_to_image(image, bbox):
    (x, y, w, h) = bbox
    start_point = (x, y)
    end_point = (x + w, y + h)
    cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)


def add_text_to_image(image, text, x, y):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), int(1.5))


def add_detection_boxes_to_image(image_array, detections):
    for detection in detections:
        add_rectangle_to_image(image_array, detection["bbox"])
        add_text_to_image(image_array, detection["class"], detection["bbox"][0], detection["bbox"][1] - 5)
    return image_array


engine = create_engine('sqlite:///library.db')
connector = DBSynchronizer("dataset", engine)

print("Synchronizing database...")
connector.synchronize()

analyzer = DBAnalyzer(engine)

print("Clustering images in database...")
clustered_paths = analyzer.hierarchical_clustering()
print(f"Found {len(clustered_paths)} clusters")
for i, paths in enumerate(clustered_paths):
    montage = None
    for p in paths[:10]:
        image = cv2.imread(p)
        image = cv2.resize(image, (150, 150))
        if montage is None:
            montage = image
        else:
            montage = np.hstack([montage, image])
    cv2.imshow(f"Cluster {i}", montage)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Finding images with clock...")
db_reader = DBReader(engine)
paths = db_reader.get_files_with_object(["clock"])
print(f"Found {len(paths)} images with clock")

for p in paths[:10]:
    _, filename = os.path.split(p)
    image = cv2.imread(p)
    detections = db_reader.get_detections(p)
    clock_detections = [d for d in detections if d["class"] == "clock"]
    image = add_detection_boxes_to_image(image, clock_detections)
    cv2.imshow(f"{filename}", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Making image query...")
file_path = "dataset/124002.png"
query_image = load_image(file_path)
paths, scores = analyzer.image_query(query_image)
cv2.imshow(f"query image", pil_to_array(query_image))
for (p, score) in zip(paths, scores):
    _, filename = os.path.split(p)
    image = cv2.imread(p)
    image = cv2.resize(image, (500, 500))
    cv2.imshow(f"Score {score}", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
