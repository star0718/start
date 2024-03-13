import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__    #获取类的名字
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)     #按正态分布对(m.weight.data)随机赋值。
        if hasattr(m, "bias") and m.bias is not None:       #如果m包含bias同时m.bias不为空
            torch.nn.init.constant_(m.bias.data, 0.0)       #将m.bias.data赋值为0.0
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)     #按正态分布对(m.weight.data)随机赋值。
        torch.nn.init.constant_(m.bias.data, 0.0)       #将m.bias.data赋值为0.0


##############################
#           RESNET
##############################

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()     #初始化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)     #定义平均池化层为自适应平均池化
        self.fc = nn.Sequential(    #序列容器，利用nn.Sequential()搭建好模型架构，
            # 模型前向传播时调用forward()方法，模型接收的输入首先被传入nn.Sequential()包含的第一个网络模块中。
            # 然后，第一个网络模块的输出传入第二个网络模块作为输入，
            # 按照顺序依次计算并传播，直到nn.Sequential()里的最后一个模块输出结果。

            nn.Linear(channel, channel // reduction, bias=False),   #神经网络的线性层
            nn.ReLU(inplace=True),  #inplace=True将计算得到的值直接覆盖之前的值
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

            #残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),  #类似于镜像填充。
            nn.Conv2d(in_features, in_features, 3),     #对由多个输入平面组成的输入信号进行二维卷积
            nn.InstanceNorm2d(in_features),     #在像素上对HW做归一化
            nn.ReLU(inplace=True),          #改变输入的数据
            nn.ReflectionPad2d(1),      #类似于镜像填充。
            nn.Conv2d(in_features, in_features, 3),     #对由多个输入平面组成的输入信号进行二维卷积
            nn.InstanceNorm2d(in_features),     #在像素上对HW做归一化
        )

    def forward(self, x):
        return x + self.block(x)

        #生成器残差网络
class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()     #初始化

        channels = input_shape[0]

        # Initial convolution block 初始卷积块
        out_features = 64   #初始化输出的特征数为64
        model = [
            nn.ReflectionPad2d(channels),   #镜像补充
            nn.Conv2d(channels, out_features, 7),   #对由多个输入平面组成的输入信号进行二维卷积
            nn.InstanceNorm2d(out_features),        #在像素上对HW做归一化
            nn.ReLU(inplace=True),      #改变输入的数据
        ]

        in_features = out_features

        model += [SELayer(channels)]

        # Downsampling  下采样
        for _ in range(2):
            out_features *= 2   #下采样特征数*2加倍
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),   #对由多个输入平面组成的输入信号进行二维卷积
                nn.InstanceNorm2d(out_features),    #在像素上对HW做归一化
                nn.ReLU(inplace=True),      #改变输入的数据
            ]
            in_features = out_features

        # Residual blocks残差块
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling    上采样
        for _ in range(2):
            out_features //= 2  #上采样特征数/2
            model += [
                nn.Upsample(scale_factor=2),    #图像高度/宽度/深度的乘数
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),   #对由多个输入平面组成的输入信号进行二维卷积
                nn.InstanceNorm2d(out_features),    #在像素上对HW做归一化
                nn.ReLU(inplace=True),      #改变输入的数据
            ]
            in_features = out_features

        # Output layer  输出层
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################

        #判别器
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()       #初始化生成器

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
                #返回每个鉴别器块的下采样层
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))   #层添加对HW做归一化处理后的结果
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)), #上下各添加1dim
            nn.Conv2d(512, 1, 4, padding=1) #卷积扩张
        )

    def forward(self, img):
        return self.model(img)
