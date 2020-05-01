import io
import json
import os
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def get_file_paths(dir_path: str, file_extensions: Tuple[str, str]):
    file_paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path)
                  if name.endswith(file_extensions)]
    return file_paths


def load_json(file_path: str):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data


def save_json(file_path: str, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)


def is_image_valid(image):
    n_dimensions = len(np.array(image).shape)
    if n_dimensions != 3:
        return False
    n_channels = np.array(image).shape[2]
    if n_channels != 3:
        return False
    return True


def load_image(file_path):
    with open(file_path, 'rb') as f:
        try:
            image = Image.open(io.BytesIO(f.read()))
        except OSError:
            image = None

    is_valid = is_image_valid(image)
    if is_valid:
        return image
    else:
        return None


def pil_to_array(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
