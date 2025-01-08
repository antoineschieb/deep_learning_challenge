from PIL import Image
import torch
from torch.utils.data import Dataset

from constants import labels_map


class CustomImageDataset(Dataset):
    def __init__(self, samples, load_embeddings=False, transform=None):
        self.samples = samples
        self.load_embeddings = load_embeddings
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]  # Allow slicing

        data_index = self.samples[idx]
        label = data_index // 100
        class_sample = data_index % 100
        
        if self.load_embeddings:
            emb_path = f"challenge/{labels_map[label]}/c{label+1}_{str(class_sample).zfill(3)}.pt"
            emb = torch.load(emb_path)
            return emb, label
        else:
            img_path = f"challenge/{labels_map[label]}/c{label+1}_{str(class_sample).zfill(3)}.tif"
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            return image, label
    