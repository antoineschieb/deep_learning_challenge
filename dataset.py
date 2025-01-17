from functools import cache
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot

from constants import LABELS_MAP


class CustomImageDataset(Dataset):
    """
    Custom Dataset class that loads either images or embeddings.
    There are 4 classes of 100 samples each, so each sample is associated to an integer between 0 and 399. 

    Args:
    - samples (List or slice): Sample to use for this dataset instance.
    
    - load_embeddings (bool): Wether to load embeddings or images. 
      If load_embeddings=False, then images_folder_path needs to be specified.
      If load_embeddings=True, then the embeddings need to be present in the embeddings/ folder.
      This can be done by running the create_embeddings.py script first

    - images_folder_path (string): Path to the folder containing the challenge images
    """
    def __init__(self, samples, load_embeddings=False, images_folder_path=None):
        if not load_embeddings and images_folder_path is None:
            raise RuntimeError('You must provide the image folder path if you wish to load images')
        self.samples = samples
        self.load_embeddings = load_embeddings
        self.images_folder_path = images_folder_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]  # Allow slicing

        data_index = self.samples[idx]
        int_label = data_index // 100
        label = one_hot(torch.tensor(int_label), num_classes=4).to(torch.float)
        class_sample = data_index % 100
        
        if self.load_embeddings:
            emb_path = Path.cwd() / "embeddings" / LABELS_MAP[int_label] / f"c{int_label+1}_{str(class_sample).zfill(3)}.pt"
            emb = torch.load(emb_path, weights_only=True)
            return emb, label
        else:
            img_path = Path(self.images_folder_path) / LABELS_MAP[int_label] / f"c{int_label+1}_{str(class_sample).zfill(3)}.tif"
            image = Image.open(img_path)
            return image, label
    