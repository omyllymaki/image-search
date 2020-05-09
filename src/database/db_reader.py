import ast

from sqlalchemy.orm import sessionmaker

from src.database.models import File, Object, Detection, Features, Tag, ImageTags


class DBReader:

    def __init__(self, engine):
        Session = sessionmaker(bind=engine)
        self.session = Session()

    def get_paths(self,
                  object_names=None,
                  min_detection_confidence=None,
                  min_detection_class_score=None,
                  tags=None,
                  min_tag_confidence=None):
        query_output = self.session.query(File)

        if object_names is not None:
            query_output = query_output.join(Detection).join(Object)
        if tags is not None:
            query_output = query_output.join(ImageTags).join(Tag)
        if object_names is not None:
            query_output = query_output.filter(Object.name.in_(object_names))
        if min_detection_confidence is not None:
            query_output = query_output.filter(Detection.confidence > min_detection_confidence)
        if min_detection_class_score is not None:
            query_output = query_output.filter(Detection.class_score > min_detection_class_score)
        if tags is not None:
            query_output = query_output.filter(Tag.name.in_(tags))
        if min_tag_confidence is not None:
            query_output = query_output.filter(ImageTags.confidence > min_tag_confidence)

        query_output = query_output.all()
        return [item.absolute_path for item in query_output]

    def get_detections(self, file_path, min_confidence=None, min_class_score=None):
        query_output = self.session.query(Detection, Object, File). \
            join(Object).join(File). \
            filter(File.absolute_path == file_path)
        if min_confidence is not None:
            query_output = query_output.filter(Detection.confidence > min_confidence)
        if min_class_score is not None:
            query_output = query_output.filter(Detection.class_score > min_class_score)
        query_output = query_output.all()

        results = []
        for item in query_output:
            d = {
                "class": item.Object.name,
                "bbox": ast.literal_eval(item.Detection.bbox),
                "confidence": item.Detection.confidence,
                "class_score": item.Detection.class_score
            }
            results.append(d)
        return results

    def get_feature_vectors(self):
        query_output = self.session.query(Features, File). \
            join(File). \
            all()

        results = []
        for item in query_output:
            d = {
                "path": item.File.absolute_path,
                "feature_vector": ast.literal_eval(item.Features.feature_vector)
            }
            results.append(d)
        return results

    def get_all_tags(self):
        query_output = self.session.query(Tag.name).all()
        return [item[0] for item in query_output]

    def get_all_objects(self):
        query_output = self.session.query(Object.name).all()
        return [item[0] for item in query_output]
