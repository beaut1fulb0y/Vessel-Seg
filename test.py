import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
from PIL import Image
from torch.utils.data import DataLoader
from skimage import measure
from model import UNet
from dataset import UNetDataset

Parser = argparse.ArgumentParser()
Parser.add_argument("-b", "--batch_size", default=5, type=int, help="batch size")
Parser.add_argument("-d", "--device", default="cpu", type=str, help="device")
Parser.add_argument("-e", "--epochs", default=100, type=int, help="training epochs")
Parser.add_argument("-l", "--lr", default=0.0005, type=float, help="learning rate")
Parser.add_argument("-s", "--save_path", default="runs", type=str, help="save path")
Parser.add_argument("-w", "--num_workers", default=8, type=int, help="number of workers")

if __name__ == "__main__":
    args = Parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    elif device == "cuda":
        device = "cuda:0"

    model = UNet(1)
    param_path = os.path.join('params', 'best.pth')
    sd = torch.load(param_path, map_location=device)
    model.load_state_dict(sd)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    test_dataset = UNetDataset(20, 'data', False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    n = 0
    for batch_idx, (imgs, labels) in enumerate(test_dataloader):
        imgs, labels = imgs.to(device), labels.to(device)

        out = model(imgs)
        biseg = torch.where(out > 0.5, 255, 0)
        biseg = biseg.cpu().numpy()
        for i in range(imgs.shape[0]):
            img = np.uint8(biseg[i])
            new_img = np.zeros_like(img)
            mask = cv2.imread(os.path.join('data', 'mask', f'{n + 1}.png'))
            mask = mask[:, 0:560, :]
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            img = img[0]
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            pad = np.zeros((584, 5), dtype=np.uint8)
            masked_img = np.concatenate((masked_img, pad), axis=1)

            img = Image.fromarray(masked_img)
            img.save(os.path.join('runs', f'{n + 1}.png'))
            n += 1
