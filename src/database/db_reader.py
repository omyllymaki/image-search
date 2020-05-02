import ast

from sqlalchemy.orm import sessionmaker

from src.database.models import File, Object, Detection, Features


class DBReader:

    def __init__(self, engine):
        Session = sessionmaker(bind=engine)
        self.session = Session()

    def get_files_with_object(self, object_names):
        query_output = self.session.query(File). \
            join(Detection).join(Object). \
            filter(Object.name.in_(object_names)). \
            all()
        return [item.absolute_path for item in query_output]

    def get_detections(self, file_path):
        query_output = self.session.query(Detection, Object, File). \
            join(Object).join(File). \
            filter(File.absolute_path == file_path). \
            all()

        results = []
        for item in query_output:
            d = {
                "class": item.Object.name,
                "bbox": ast.literal_eval(item.Detection.bbox)
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
                "feature_vector": ast.literal_eval(item.Features.feature_vector)}
            results.append(d)
        return results