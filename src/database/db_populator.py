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

            detections = item["detections"]
            for detection in detections:
                class_name = detection["class"]
                bbox = detection["bbox"]
                object = get_or_create(self.session, Object, name=class_name)
                detection = get_or_create(self.session, Detection,
                                          file_id=file.id,
                                          object_id=object.id,
                                          bbox=str(bbox)
                                          )