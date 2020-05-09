import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

from src.utils import load_image


class TaggingDataset(Dataset):

    def __init__(self,
                 data,
                 transform=None,
                 classes=None):
        self.data = data
        self.transform = None
        if transform is not None:
            self.transform = transform
        if classes is not None:
            self.classes = classes
        else:
            all_tags = []
            for item in data:
                all_tags += item["tags"]
            self.classes = list(set(all_tags))
        self.one_hot_encoder = MultiLabelBinarizer(classes=self.classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        path = self.data[index]["path"]
        tags = self.data[index]["tags"]
        if not tags:
            tag_tensor = torch.zeros(len(self.classes))
        else:
            tags_one_hot_encoded = self.one_hot_encoder.fit_transform([tags]).reshape(-1)
            tag_tensor = torch.Tensor(tags_one_hot_encoded)

        image = load_image(path)

        if self.transform:
            image = self.transform(image)

        return image, tag_tensor
