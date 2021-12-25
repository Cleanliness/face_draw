import torch.nn as nn
import torch.nn.functional
import torchsummary
import math


class UnetGenerator(nn.Module):
    """
    Generator using U-net architecture
    """
    def __init__(self, imgsize):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.down1 = UnetDown(64, 128)
        self.down2 = UnetDown(128, 256)
        self.down3 = UnetDown(256, 512)

        self.up1 = UnetUp(512, 256)
        self.up2 = UnetUp(256, 128)
        self.up3 = UnetUp(128, 64)

        self.out_stack = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (7, 7)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, (7, 7)),
            nn.Sigmoid()
        )

    def forward(self, nn_input):
        """
        Forward pass on NN
        """
        stack_out = self.conv_stack(nn_input)

        d1 = self.down1(stack_out)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        x = self.up1(d3, d2)
        x = self.up2(x, d1)
        x = self.up3(x, stack_out)

        return self.out_stack(x)


class UnetUp(nn.Module):
    """
    Up-sampling layer in u-net.
    """
    def __init__(self, in_ch, out_ch):
        """
        Initialize unet.
        """
        super().__init__()
        mid_ch = out_ch + (in_ch-out_ch)//2
        print(mid_ch)

        self.conv_t = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=4, stride=2)
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=4, stride=1),
            nn.BatchNorm2d(mid_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=4, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, nn_input, skip_in):
        """
        Forward pass with direct input and input from skip connections.
        Assume skip_in channels = in_channels // 2
        Assume nn_input img dims <= skip_in dims, i.e nn_input img are smaller
        """
        # crop skip input around center to match nn_input dims
        nn_input = self.conv_t(nn_input)

        dy = skip_in.size()[2] - nn_input.size()[2]
        dx = skip_in.size()[3] - nn_input.size()[3]

        padding = (dy // 2, math.ceil(dy / 2), dx // 2, math.ceil(dx / 2))

        nn_input = torch.nn.functional.pad(nn_input, padding)

        nn_input = torch.cat((nn_input, skip_in), dim=1)
        return self.conv_stack(nn_input)


class UnetDown(nn.Module):
    """
    Down-sampling layer in u-net
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, nn_input):
        """
        Forward pass on NN
        """
        return self.stack(nn_input)


# net = UnetGenerator((200, 200))
# torchsummary.summary(net, (3, 200, 200), 200)
