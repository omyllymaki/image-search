from sqlalchemy.orm import sessionmaker

from src.database.models import File, Features, Object, Detection
from src.database.utils import get_or_create


class Populator:

    def __init__(self, engine):
        Session = sessionmaker(engine)
        self.session = Session()

    def populate(self, data):
        for item in data:
            file = get_or_create(self.session, File,
                                 absolute_path=item["absolute_path"],
                                 size=item["file_size"],
                                 last_modified=item["last_modified"],
                                 created=item["created"]
                                 )

            features = get_or_create(self.session, Features,
                                     file_id=file.id,
                                     blurriness=item["blurriness"],
                                     feature_vector=str(item["feature_vector"]),
                                     width=item["image_size"][0],
                                     height=item["image_size"][1])

            object_detections = item["object_detections"]
            for detection in object_detections:
                class_name = detection["class"]
                bbox = detection["bbox"]
                confidence = detection["object_confidence"]
                class_score = detection["class_score"]
                object = get_or_create(self.session, Object, name=class_name)
                detection = get_or_create(self.session, Detection,
                                          file_id=file.id,
                                          object_id=object.id,
                                          bbox=str(bbox),
                                          confidence=confidence,
                                          class_score=class_score
                                          )

            face_detections = item["face_detections"]
            for detection in face_detections:
                bbox = detection["bbox"]
                confidence = detection["confidence"]
                object = get_or_create(self.session, Object, name="face")
                detection = get_or_create(self.session, Detection,
                                          file_id=file.id,
                                          object_id=object.id,
                                          bbox=str(bbox),
                                          confidence=confidence
                                          )
