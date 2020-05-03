import os

CONFIG_PATH = os.path.join('config', 'yolov3.cfg')
WEIGHTS_PATH = os.path.join('config', 'yolov3.weights')
CLASSES_PATH = os.path.join('config', 'coco.names')
MODEL_IMAGE_SIZE = 608
CONFIDENCE_THRESHOLD = 0.8
SUPRESSION_THRESHOLD = 0.4
CLASS_SCORE_THRESHOLD = 0.0
DEVICE = 'cuda'  # 'cuda' or 'cpu'
