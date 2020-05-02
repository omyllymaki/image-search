import sys

from sqlalchemy import create_engine

from src.database.models import create_tables
from src.database.db_populator import Populator

sys.path.append('./src/object_detection/PyTorchYOLOv3')

from constants import EXTENSIONS
from src.image_processor import ImageProcessor
from src.utils import recursive_glob, load_image, collect_file_info

processor = ImageProcessor()
paths = recursive_glob("small_dataset", EXTENSIONS)[:15]

data = []
for i, path in enumerate(paths):
    print(f"{i + 1}/{len(paths)}")
    image = load_image(path)
    results = processor.process(image)
    file_info = collect_file_info(path)
    data.append({**results, **file_info})

engine = create_engine('sqlite:///library.db')
create_tables(engine)
populator = Populator(engine)
populator.populate(data)
