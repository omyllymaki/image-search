import torch

from src.tagging.image_transforms import inference_transforms
from src.tagging.models import get_tagger


class Tagger:

    def __init__(self, threshold=0.3, device="cuda:0"):
        self.device = device
        with open('classes.txt', 'r') as f:
            classes = f.readlines()
        self.classes = [line.rstrip('\n') for line in classes]
        self.model = get_tagger(len(self.classes))
        self.model.load_state_dict(torch.load("tagger.model"))
        self.model.eval()
        self.model.to(self.device)
        self.threshold = threshold

    def run(self, image):
        image_tensor = inference_transforms(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        output = self.model(image_tensor)
        probabilities = torch.exp(output.cpu()).detach().numpy().reshape(-1)

        results = []
        for i, p in enumerate(probabilities):
            if p >= self.threshold:
                results.append({
                    "tag": self.classes[i],
                    "confidence": p
                })
        return results
