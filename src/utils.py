import io
import json
import os
import stat
import time
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


def recursive_glob(rootdir, suffix):
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


def load_images(paths):
    valid_paths, images = [], []
    for path in paths:
        image = load_image(path)
        if image is not None:
            images.append(image)
            valid_paths.append(path)
    return images, valid_paths


def collect_file_info(path):
    filename = os.path.split(path)[-1]
    absolute_path = os.path.abspath(path)
    file_stats = os.stat(path)
    file_info = {
        'path': path,
        'absolute_path': absolute_path,
        'filename': filename,
        'file_size': file_stats[stat.ST_SIZE],
        'last_modified_time': time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime(file_stats[stat.ST_MTIME])),
        'creation_time': time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime(file_stats[stat.ST_CTIME]))
    }
    return file_info
