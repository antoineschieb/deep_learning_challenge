import os
from functools import cache
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot

from constants import DATA_ROOT, LABELS_MAP


class CustomImageDataset(Dataset):
    def __init__(self, samples, load_embeddings=False):
        self.samples = samples
        self.load_embeddings = load_embeddings

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]  # Allow slicing

        data_index = self.samples[idx]
        label = data_index // 100
        class_sample = data_index % 100
        
        if self.load_embeddings:
            emb_path = Path(Path.cwd() / "embeddings" / LABELS_MAP[label] / f"c{label+1}_{str(class_sample).zfill(3)}.pt")
            emb = torch.load(emb_path, weights_only=True)
            return emb, one_hot(torch.tensor(label), num_classes=4).to(torch.float)
        else:
            img_path = Path(DATA_ROOT / LABELS_MAP[label] / f"c{label+1}_{str(class_sample).zfill(3)}.tif")
            image = Image.open(img_path)
            return image, one_hot(torch.tensor(label), num_classes=4).to(torch.float)
        
    