import torch
from transformers import AutoImageProcessor, AutoModel
from dataset import CustomImageDataset
from torchvision.transforms import Resize
from tqdm import tqdm

from constants import labels_map


if __name__ == "__main__":

    # Load phikon-v2 encoder
    processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    model = AutoModel.from_pretrained("owkin/phikon-v2")
    model.eval()

    full_ds = CustomImageDataset(list(range(400)), transform=Resize(size=(224,224)))

    with torch.inference_mode():
        for i,(image,label) in tqdm(enumerate(full_ds)):
            
            # Process the image
            inputs = processor(image, return_tensors="pt")
            # Get the features
        
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # (1, 1024) shape

            class_sample = i % 100

            torch.save(features, f"challenge/{labels_map[label]}/c{label+1}_{str(class_sample).zfill(3)}.pt")