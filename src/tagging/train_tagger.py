import argparse

import numpy as np
import pandas as pd
import torch

from src.tagging.data_loader import DeviceDataLoader
from src.tagging.image_transforms import training_transforms, inference_transforms
from src.tagging.models import get_tagger
from src.tagging.tagging_dataset import TaggingDataset
from src.tagging.utils import get_device

BATCH_SIZE = 6
EPOCHS = 5
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.01
PROBABILITY_THRESHOLD = 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to labels file.")
    parser.add_argument("-w", "--weights_output", default="./tagger.model", help="Path of output model weights.")
    parser.add_argument("-c", "--classes_output", default="./classes.txt", help="Path of output classes names.")
    args = parser.parse_args()

    df = pd.read_csv(args.input, header="infer", engine='python')
    df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: x.split(","))
    df = df.sample(frac=1).reset_index(drop=True)

    n_samples = df.shape[0]
    train_indices = np.arange(int(0.7 * n_samples))
    valid_indices = np.arange(int(0.7 * n_samples), n_samples)
    df_train = df.loc[train_indices]
    df_valid = df.loc[valid_indices]

    device = get_device()
    ds_train = TaggingDataset(df_train, transform=training_transforms)
    dl_train = DeviceDataLoader(ds_train, device, batch_size=6, shuffle=True, num_workers=4)

    ds_valid = TaggingDataset(df_valid, transform=inference_transforms, classes=ds_train.classes)
    dl_valid = DeviceDataLoader(ds_valid, device, batch_size=6, shuffle=True, num_workers=4)

    model = get_tagger(len(ds_train.classes))
    model.to(device)
    loss_function = torch.nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        model.train()
        n_correct, n_all = 0, 0
        losses = []
        for i_batch, sample in enumerate(dl_train):
            images, tags = sample
            output = model(images)
            loss = loss_function(output, tags)
            losses.append(loss.item())

            y_true = tags.cpu().detach().numpy().astype(int)
            probabilities = torch.exp(output.cpu()).detach().numpy()
            y_pred = (probabilities > PROBABILITY_THRESHOLD).astype(int)

            n_correct += np.sum(y_true == y_pred)
            n_all += y_true.size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}; batch {i_batch}, training loss {loss.item()}")

        training_accuracy = 100 * n_correct / n_all
        training_avg_loss = np.mean(losses)
        print(f"Epoch {epoch + 1}; training accuracy {training_accuracy:0.2f} %")
        print(f"Epoch {epoch + 1}; average training loss {training_avg_loss:0.3f}")

        model.eval()
        with torch.no_grad():
            n_correct, n_all = 0, 0
            losses = []
            for i_batch, sample in enumerate(dl_valid):
                images, tags = sample
                output = model(images)
                loss = loss_function(output, tags)
                losses.append(loss.item())

                y_true = tags.cpu().detach().numpy().astype(int)
                probabilities = torch.exp(output.cpu()).detach().numpy()
                y_pred = (probabilities > PROBABILITY_THRESHOLD).astype(int)

                n_correct += np.sum(y_true == y_pred)
                n_all += y_true.size

        valid_accuracy = 100 * n_correct / n_all
        valid_avg_loss = np.mean(losses)
        print(f"Epoch {epoch + 1}; valid accuracy {valid_accuracy:0.2f} %")
        print(f"Epoch {epoch + 1}; average valid loss {valid_avg_loss:0.3f}")

    print("Training done!")

    print(f"Saving model weighs to path {args.weights_output}")
    torch.save(model.state_dict(), args.weights_output)

    print(f"Saving class names to path {args.classes_output}")
    with open(args.classes_output, 'w') as f:
        for item in ds_train.classes:
            f.write("%s\n" % item)


if __name__ == "__main__":
    main()
