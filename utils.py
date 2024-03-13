import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np

from torchvision.utils import save_image


class ReplayBuffer:
    def __init__(self, max_size=50):    #初始化
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        #判断输入的max_size是否小于0，如果输入的max_size<=0输出Empty buffer or trying to create a black hole. Be careful.
        self.max_size = max_size    #输入的最大长度为50
        self.data = []  #输入的

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)   #扩展维度，返回一个新的张量，对输入的既定位置插入维度0
            if len(self.data) < self.max_size:
                self.data.append(element)   #添加元素element
                to_return.append(element)   #添加元素element
            else:
                if random.uniform(0, 1) > 0.5:  #random.uniform(0,1)0和1之间随机生成的浮点数
                    i = random.randint(0, self.max_size - 1)    #i是从0-self.max_size - 1之间随机生成的整数
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
