from sqlalchemy.orm import sessionmaker

from src.database.models import File, Features, Object, Detection, Tag, ImageTags


class Populator:

    def __init__(self, engine):
        Session = sessionmaker(engine)
        self.session = Session()

    def populate(self, item):
        file = self.get_or_create(File,
                                  absolute_path=item["absolute_path"],
                                  size=item["file_size"],
                                  last_modified=item["last_modified"],
                                  created=item["created"]
                                  )

        features = self.get_or_create(Features,
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
            object = self.get_or_create(Object, name=class_name)
            detection = self.get_or_create(Detection,
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
            object = self.get_or_create(Object, name="face")
            detection = self.get_or_create(Detection,
                                           file_id=file.id,
                                           object_id=object.id,
                                           bbox=str(bbox),
                                           confidence=confidence,
                                           class_score=1.0
                                           )

        tags = item["tags"]
        for item in tags:
            tag_name = item["tag"]
            confidence = item["confidence"]
            tag = self.get_or_create(Tag, name=tag_name)
            tag_detection = self.get_or_create(ImageTags,
                                               file_id=file.id,
                                               tag_id=tag.id,
                                               confidence=confidence,
                                               )

    def get_or_create(self, model, **kwargs):
        instance = self.session.query(model).filter_by(**kwargs).first()
        if instance:
            return instance
        else:
            instance = model(**kwargs)
            self.session.add(instance)
            self.session.commit()
            return instance
