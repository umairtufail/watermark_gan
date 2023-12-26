import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return sum([len(os.listdir(os.path.join(self.root_dir, cls))) for cls in self.classes])

    def __getitem__(self, index):
        cls_index = index % len(self.classes)
        cls = self.classes[cls_index]
        img_file = np.random.choice(os.listdir(os.path.join(self.root_dir, cls)))
        img_path = os.path.join(self.root_dir, cls, img_file)
        image = np.array(Image.open(img_path).convert("RGB"))
        split_half = int(image.shape[1] / 2)
        input_image = image[:, :split_half, :]
        target_image = image[:, split_half:, :]

        # Apply transformations
        if self.transform:
            input_image = self.transform(Image.fromarray(input_image))
            target_image = self.transform(Image.fromarray(target_image))

        return input_image, target_image
