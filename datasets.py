import glob #该模块用来查找文件目录和文件，并将搜索的到的结果返回到一个列表中
import random   #该模块提供的便是操作系统相关的功能了，可以处理文件和目录
import os

from torch.utils.data import Dataset
from PIL import Image   #图像裁剪所用到的库
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)    #创建一个RGB类型的和输入图像大小相同的空图像
    rgb_image.paste(image)  #将输入的图像的内容复制到创建的空图上
    return rgb_image


class ImageDataset(Dataset):    # 继承Dataset，复写__getitem__和__len__
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        # 排序
        self.files_A = sorted(glob.glob(os.path.join(root, "%s" % mode + "A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s" % mode + "B") + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])   #打开一个图片

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
