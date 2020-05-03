import os

import cv2
import numpy as np
from sqlalchemy import create_engine

from src.database.db_reader import DBReader
from src.db_synchronizer import DBSynchronizer
from src.duplicate_detector import DuplicateDetector
from src.image_clustering import ImageClustering
from src.image_search import ImageSearch
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
db_reader = DBReader(engine)

print("Synchronizing database...")
connector.synchronize()

print("Finding images with detected clock...")
paths = db_reader.get_files_with_object(["clock"])
print(f"Found {len(paths)} images with clock")
for p in paths:
    _, filename = os.path.split(p)
    image = cv2.imread(p)
    detections = db_reader.get_detections(p)
    clock_detections = [d for d in detections if d["class"] == "clock"]
    image = add_detection_boxes_to_image(image, clock_detections)
    cv2.imshow(f"{filename}", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Reading feature array from database")
data = db_reader.get_feature_vectors()
feature_array = [d["feature_vector"] for d in data]
all_paths = [d["path"] for d in data]
searcher = ImageSearch(feature_array)

print("Making image query...")
file_path = "dataset/124002.png"
query_image = load_image(file_path)
indices, scores = searcher.query(query_image)
print(f"Found {len(indices)} matches")
search_result_paths = [all_paths[i] for i in indices]
cv2.imshow(f"query image", pil_to_array(query_image))
for (p, score) in zip(search_result_paths, scores):
    _, filename = os.path.split(p)
    image = cv2.imread(p)
    image = cv2.resize(image, (500, 500))
    cv2.imshow(f"Score {score:0.2f}", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Clustering images in database...")
clustering = ImageClustering(feature_array)
clustering.show_dendogram()
distance = float(input('Enter threshold distance for clustering:'))
clusters = clustering.hierarchical_clustering(distance=distance)

clustered_paths = {}
for i, c in enumerate(clusters):
    if clustered_paths.get(c) is None:
        clustered_paths[c] = []
    clustered_paths[c].append(all_paths[i])

print(f"Found {len(clustered_paths)} clusters")
for c, paths in clustered_paths.items():
    montage = None
    for p in paths[:10]:
        image = cv2.imread(p)
        image = cv2.resize(image, (150, 150))
        if montage is None:
            montage = image
        else:
            montage = np.hstack([montage, image])
    cv2.imshow(f"Cluster {c}", montage)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Finding duplicate images...")
duplicate_detector = DuplicateDetector(feature_array)
duplicates = duplicate_detector.find_duplicates(0.98)

for ref_index, duplicate_indices in duplicates.items():

    if len(duplicate_indices) > 0:
        ref_filename = os.path.split(all_paths[ref_index])[1]
        duplicate_filenames = [os.path.split(all_paths[i])[1] for i in duplicate_indices]
        print(f"Photo {ref_filename}; duplicates {duplicate_filenames}")

        reference_image = cv2.imread(all_paths[ref_index])
        duplicate_images = cv2.resize(reference_image, (150, 150))

        for i in duplicate_indices:
            p = all_paths[i]
            image = cv2.imread(p)
            image = cv2.resize(image, (150, 150))
            duplicate_images = np.hstack([duplicate_images, image])

            cv2.imshow(f"Duplicate images for {ref_filename}", duplicate_images)

cv2.waitKey(0)
cv2.destroyAllWindows()
