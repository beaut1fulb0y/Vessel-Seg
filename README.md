# Vessel-Seg
 Retinal blood vessel segmentation by UNet approach

## 模型安装

### 库依赖

- torch
- tensorboard
- scikit-learn
- scikit-image
- numpy
- libtiff

从github上克隆项目到本地，然后将DRIVE数据放到根目录下，并且将数据文件夹命名为`data`。数据集的结构应该如下：
- data
    - images
    - manual
    - mask

然后分别运行`convert.py`和`delete.py`，两个文件的作用分别是将所有图像转化为png格式，以及将原图删去。

## 模型说明

本项目利用UNet实现，训练可以运行`main.py`，可以通过一下命令行实现：

```bash
python main.py
```

如果有cuda或者cudnn，可以使用以下命令行：

```bash
python main.py -d cuda
```

测试过程分为两步：

1. 首先运行`test.py`，生成在测试集上的分割图像，其保存位置在`runs/`目录下

2. 运行`calc.py`，计算TPR，FPR等参数，并且绘制ROC曲线，ROC曲线图像保存在根目录下
