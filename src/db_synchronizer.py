import os
import sys

sys.path.append('./src/object_detection/PyTorchYOLOv3')

from src.database.models import create_tables, File
from src.database.db_populator import Populator

from src.image_processor import ImageProcessor
from sqlalchemy.orm import sessionmaker
from src.utils import recursive_glob, load_image, collect_file_info


class DBSynchronizer:

    def __init__(self, images_folder, db_engine, image_types=(".jpg", ".png")):
        self.images_folder = images_folder
        self.image_types = image_types
        self.engine = db_engine
        self._initialize_database()
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def synchronize(self):
        local_paths = self._get_image_paths()
        db_paths = self._get_image_paths_in_database()
        local_paths_not_in_db = set(local_paths) - set(db_paths)
        if len(local_paths_not_in_db) > 0:
            self._add_new_images_to_db(local_paths_not_in_db)
        db_paths_not_in_local_paths = set(db_paths) - set(local_paths)
        if len(db_paths_not_in_local_paths) > 0:
            self._remove_images_from_db(db_paths_not_in_local_paths)

    def _add_new_images_to_db(self, paths):
        print(f"Found {len(paths)} new images that are not in database")
        print("Creating new items to database...")
        populator = Populator(self.engine)
        self.processor = ImageProcessor()
        for i, path in enumerate(paths):
            print(f"{i + 1}/{len(paths)}")
            image = load_image(path)
            results = self.processor.process(image)
            file_info = collect_file_info(path)
            data = {**results, **file_info}
            populator.populate(data)

    def _remove_images_from_db(self, paths):
        # TODO: delete also related rows from other tables
        print(f"Found {len(paths)} images from database that do not exist")
        print("Deleting non-existing items from database...")
        items = self.session.query(File).filter(File.absolute_path.in_(paths))
        items.delete(synchronize_session=False)
        self.session.commit()

    def _get_image_paths(self):
        paths = recursive_glob(self.images_folder, self.image_types)
        return [os.path.abspath(path) for path in paths]

    def _initialize_database(self):
        create_tables(self.engine)

    def _get_image_paths_in_database(self):
        items = self.session.query(File).all()
        paths = [item.absolute_path for item in items]
        return paths
