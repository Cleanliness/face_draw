import torch
import torch.nn as nn
import torchsummary


class PatchGAN(nn.Module):
    """
    Discriminator network outputting real/fake values
    in 'patches' instead of as a single scalar value (baseline).
    """

    def __init__(self):
        """
        Initialize PatchGAN.
        """
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(6, 64, (4, 4), stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, (4, 4), stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, (4, 4), stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 1, (4, 4), stride=1),
            nn.Sigmoid()
        )

    def forward(self, nn_in):
        """
        Forward pass on patchGAN
        """
        return self.conv_stack(nn_in)


net = PatchGAN()
torchsummary.summary(net, (6, 200, 200), 64)


