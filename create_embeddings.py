import sys
import torch
from transformers import AutoImageProcessor, AutoModel
from dataset import CustomImageDataset
from tqdm import tqdm
from pathlib import Path

from constants import LABELS_MAP


if __name__ == "__main__":
    # Create the folder containing the embeddings, replicating the same structure as the folder containing the images.
    for c in range(1,5):
        Path(Path.cwd() / "embeddings" / f"class{c}").mkdir(parents=True, exist_ok=True)

    # Load phikon-v2 encoder
    processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    model = AutoModel.from_pretrained("owkin/phikon-v2")
    model.eval()

    # Create a dataset with all the samples
    full_ds = CustomImageDataset(list(range(400)), load_embeddings=False, images_folder_path=sys.argv[1])

    with torch.inference_mode():
        for i,(image,label) in tqdm(enumerate(full_ds)):
            # Process the image
            inputs = processor(image, return_tensors="pt")
            # Get the features
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # (1, 1024) shape

            class_sample = i % 100
            int_label = int(torch.argmax(label))
            
            emb_path = Path.cwd() / "embeddings" / LABELS_MAP[int_label] / f"c{int_label+1}_{str(class_sample).zfill(3)}.pt"
            torch.save(features, emb_path)
    