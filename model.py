import torch
import torch.nn as nn


class TransformerNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.main = nn.Sequential(
        conv_layer(3, 32, 9, 1),
        conv_layer(32, 64, 3, 2),
        conv_layer(64, 128, 3, 2),
        ResBlock(128, 3),
        ResBlock(128, 3),
        ResBlock(128, 3),
        ResBlock(128, 3),
        DeConv(128, 64, 3, 2, 1),
        nn.ReLU(),
        DeConv(64, 32, 3, 2, 1),
        nn.ReLU(),
        conv_layer(32, 3, 9, 1)[:-2]
    )

  def forward(self, x):
    return self.main(x)



def conv_layer(in_ch, out_ch, kernel_size, stride):
    rp = kernel_size // 2
    return nn.Sequential(
        nn.ReflectionPad2d(rp),
        nn.Conv2d(in_ch, out_ch, kernel_size, stride),
        nn.InstanceNorm2d(out_ch, affine = True),
        nn.ReLU()
    )


class ResBlock(nn.Module):

    def __init__(self, channels = 128, kernel_size = 3):
        super().__init__()
        self.conv1 = conv_layer(channels, channels, kernel_size, stride = 1)
        self.conv2 = conv_layer(channels, channels, kernel_size, stride = 1)[:-1]
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class DeConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, out_padding):
        super().__init__()
        padding_size = kernel_size // 2
        self.convT = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding_size, out_padding)
        self.norm = nn.InstanceNorm2d(out_ch, affine = True)

    def forward(self, x):
        return self.norm(self.convT(x))
