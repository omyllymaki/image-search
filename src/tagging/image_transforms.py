from torchvision import transforms

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
IMAGE_NET_IMAGE_SIZE = 224

training_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(1.0, 1.0)),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(size=IMAGE_NET_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
])

inference_transforms = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=IMAGE_NET_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
])
