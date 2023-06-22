import os
import cv2
from PIL import Image, ImageSequence
from libtiff import TIFF

for i in range(1, 41):
    if i < 10:
        # path = os.path.join("mask", f"0{i}_mask.gif")
        y_path = os.path.join("manual", f"0{i}_manual1.gif")
        x_path = os.path.join("images", f"0{i}_test.tif")

    else:
        # path = os.path.join("mask", f"{i}_mask.gif")
        y_path = os.path.join("manual", f"{i}_manual1.gif")
        if i > 20:
            x_path = os.path.join("images", f"{i}_training.tif")
        else:
            x_path = os.path.join("images", f"{i}_test.tif")

    # im = Image.open(path)
    y = Image.open(y_path)
    img = TIFF.open(x_path)
    x = img.read_image()
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    # for frame in ImageSequence.Iterator(im):
    #     frame.save(os.path.join("mask", "{i}.png"))
    for frame in ImageSequence.Iterator(y):
        frame.save(os.path.join("manual", f"{i}.png"))
    cv2.imwrite(os.path.join("images", f"{i}.png"), x)
