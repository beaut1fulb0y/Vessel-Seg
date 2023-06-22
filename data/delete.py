import os

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
    # os.remove(path=path)
    os.remove(path=y_path)
    os.remove(path=x_path)
