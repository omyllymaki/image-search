import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

from src.utils import load_image


class TaggingDataset(Dataset):

    def __init__(self,
                 df,
                 transform=None,
                 classes=None):
        self.df = df
        self.unique_tags = self.df.iloc[:, 1].explode().unique().tolist()
        self.transform = None
        if transform is not None:
            self.transform = transform
        if classes is not None:
            self.classes = classes
        else:
            self.classes = self.df.iloc[:, 1].explode().unique().tolist()
        self.one_hot_encoder = MultiLabelBinarizer(classes=self.classes)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):

        tags = self.df.iloc[index][1]
        tags_one_hot_encoded = self.one_hot_encoder.fit_transform([tags]).reshape(-1)
        tag_tensor = torch.Tensor(tags_one_hot_encoded)

        path = self.df.iloc[index][0]
        image = load_image(path)

        if self.transform:
            image = self.transform(image)

        return image, tag_tensor