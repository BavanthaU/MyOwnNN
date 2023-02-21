import torch
import torch.nn as nn
from torchsummary import summary


# in_channels (int) – Number of channels in the input image out_channels (int) – Number of channels produced by the
# convolution : forwards plays the same role as __call__ does for a regular python class. Basically when you run model(
# input) this calls internally forward + some extra code around this function to add functionalities.

# As you can see in '__init__' function we designed the model, in 'forward' function we specified the data flow.
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size=3, pad=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernal_size, padding=pad)
        # in channel is all about how many layers in the input image and out = how many output layers
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernal_size, padding=pad)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=2, stride=2, pad=0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=pad)
        self.conv = ConvBlock(out_channel + out_channel, out_channel)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)  # [-1, 512, 64, 64] 2 nd dim is the number of filters
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        # ENCODER
        self.e1 = EncoderBlock(3, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        # BOTTLENECK
        self.b = ConvBlock(512, 1024)

        # Decoder
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        # CLASSIFIER
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        # Encoder
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # Bottleneck
        b = self.b(p4)

        # Decoder
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        # Classifier
        outputs = self.outputs(d4)

        return outputs


unet = Unet()
summary(unet, (3, 512, 512))
