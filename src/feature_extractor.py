import torch
from torchvision import models as models
from torchvision import transforms


class FeatureExtractor:
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NET_STD = [0.229, 0.224, 0.225]
    IMAGE_NET_IMAGE_SIZE = 224
    DEVICE = "cuda:0"

    TRANSFORMATIONS = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=IMAGE_NET_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_NET_MEAN, IMAGE_NET_STD)
    ])

    def __init__(self):
        self.feature_vector = torch.Tensor().to(self.DEVICE)
        self._prepare_model()

    def _copy_data(self, model, input, output):
        output_data = output.data.view(-1)
        self.feature_vector.resize_(output_data.shape)
        self.feature_vector.copy_(output_data)

    def _prepare_model(self):
        model = models.resnet18(pretrained=True)
        model.to(self.DEVICE)
        model.eval()
        feature_layer = model._modules.get('avgpool')
        feature_layer.register_forward_hook(self._copy_data)
        self.model = model

    def extract(self, image):
        transformed_image = self.TRANSFORMATIONS(image).unsqueeze(0).to(self.DEVICE)
        self.model(transformed_image)
        return self.feature_vector.detach().cpu().numpy().tolist()
