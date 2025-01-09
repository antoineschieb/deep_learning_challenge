import sys
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

from constants import LABELS_MAP
from model import MLP


def classify(image, processor, encoder, mlp):
    with torch.inference_mode():
        # Process the image
        inputs = processor(image, return_tensors="pt")
        # Get the features
        outputs = encoder(**inputs)
        features = outputs.last_hidden_state[:, 0, :]  # (1, 1024) shape
        # Find predicted class
        pred_int = int(torch.argmax(mlp(features)))

    return LABELS_MAP[pred_int]


if __name__ == '__main__':
    # Load encoder 
    processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    encoder = AutoModel.from_pretrained("owkin/phikon-v2")
    encoder.eval()

    # Load  MLP
    mlp = MLP(d_in=1024, d_hidden=128, d_out=4)
    mlp.load_state_dict(torch.load('mlp.pth', weights_only=True))
    mlp.eval()

    image = Image.open(sys.argv[1])
    print(f'Prediction: {classify(image, processor, encoder, mlp)}')

