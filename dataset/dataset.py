import os
import numpy as np
import torch.nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UNetDataset(Dataset):
    def __init__(self, l, path, train):
        super(UNetDataset, self).__init__()
        self.len = l
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])
        self.train = train
        self.data_path = path

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        path_to_image = os.path.join(self.data_path, 'images', f"{item + 1 if not self.train else item + 21}.png")
        path_to_label = os.path.join(self.data_path, 'manual', f"{item + 1 if not self.train else item + 21}.png")
        img = Image.open(path_to_image)
        img = self.transform(img)
        img = img[:, :, 0:560]
        label = Image.open(path_to_label)
        label = self.transform(label)
        label = label[:, :, 0:560]
        label = torch.where(label > 0.5, 1, 0)
        # label = label.squeeze(0)
        # label_onehot = F.one_hot(label)
        # label_onehot = label_onehot.permute(2, 0, 1).contiguous().float()
        return img, label


if __name__ == "__main__":
    img_path = os.path.join('..', 'data', 'images', '1.png')
    img = Image.open(img_path)
    img.show()

    m = UNetDataset(20, '../data', False)
    img, label = m[39]
    img.show()
    print("")