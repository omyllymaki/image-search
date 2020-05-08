from torch.utils.data import DataLoader

from src.tagging.utils import to_device


class DeviceDataLoader:
    def __init__(self, ds, device, *args, **kwargs):
        self.dl = DataLoader(ds, *args, **kwargs)
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)
