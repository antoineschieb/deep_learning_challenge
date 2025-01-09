import torch
from transformers import AutoImageProcessor, AutoModel
from dataset import CustomImageDataset
from tqdm import tqdm

from constants import LABELS_MAP


if __name__ == "__main__":
    # Load phikon-v2 encoder
    processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    model = AutoModel.from_pretrained("owkin/phikon-v2")
    model.eval()

    full_ds = CustomImageDataset(list(range(400)))

    with torch.inference_mode():
        for i,(image,label) in tqdm(enumerate(full_ds)):
            
            # Process the image
            inputs = processor(image, return_tensors="pt")
            # Get the features
        
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # (1, 1024) shape

            class_sample = i % 100

            torch.save(features, f"challenge/{LABELS_MAP[label]}/c{label+1}_{str(class_sample).zfill(3)}.pt")