import argparse
import random

import numpy as np
import pandas as pd
import torch

from src.tagging.data_loader import DeviceDataLoader
from src.tagging.image_transforms import training_transforms, inference_transforms
from src.tagging.learner import Learner
from src.tagging.models import get_tagger
from src.tagging.tagging_dataset import TaggingDataset
from src.tagging.utils import get_device
from src.utils import load_json

BATCH_SIZE = 6
EPOCHS = 5
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.01
PROBABILITY_THRESHOLD = 0.5
TRAINING_SET_PROPORTION = 0.7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to labels JSON file.")
    parser.add_argument("-w", "--weights_output", default="./tagger.model", help="Path of output model weights.")
    parser.add_argument("-c", "--classes_output", default="./classes.txt", help="Path of output classes names.")
    args = parser.parse_args()

    data = load_json(args.input)
    random.shuffle(data)

    n_samples = len(data)
    i_split = int(TRAINING_SET_PROPORTION * n_samples)
    data_train = data[:i_split]
    data_valid = data[i_split:]

    device = get_device()
    ds_train = TaggingDataset(data_train, transform=training_transforms)
    dl_train = DeviceDataLoader(ds_train, device, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    ds_valid = TaggingDataset(data_valid, transform=inference_transforms, classes=ds_train.classes)
    dl_valid = DeviceDataLoader(ds_valid, device, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = get_tagger(len(ds_train.classes))
    model.to(device)

    learner = Learner(model,
                      learning_rate=LEARNING_RATE,
                      weight_decay=WEIGHT_DECAY,
                      probability_threshold=PROBABILITY_THRESHOLD)
    learner.learn(dl_train, dl_valid, epochs=EPOCHS)

    print("Training done!")

    print(f"Saving model weighs to path {args.weights_output}")
    learner.save_weights(args.weights_output)

    print(f"Saving class names to path {args.classes_output}")
    with open(args.classes_output, 'w') as f:
        for item in ds_train.classes:
            f.write("%s\n" % item)


if __name__ == "__main__":
    main()
