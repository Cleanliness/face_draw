import torch
import torch.nn as nn
import torchsummary


class ConvGenerator(nn.Module):
    """
    Generator NN in GAN architecture.
    """
    def __init__(self, imgsize, bottleneck_size):
        """
        Initialize generator
        :param imgsize: image dimensions as 1d tuple
        :param bottleneck_size: size of bottleneck
        """

        width = 44180
        super().__init__()
        self.bneck = bottleneck_size

        # conv2d(in_ch, filters, kernel_size)
        # padding = 0, stride = 1
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 10, (5, 5), stride=2),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.Conv2d(10, 20, (5, 5), stride=2),
            nn.BatchNorm2d(10 * 2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(width, bottleneck_size),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_size, 20, (5, 5)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(20, 3, (196, 196)),
            nn.Sigmoid()
        )

    def forward(self, nn_input):
        """
        perform one forward pass
        :param nn_input: input to the NN
        :return: output of forward pass
        """
        encoder_out = self.encoder(nn_input)
        encoder_out = torch.reshape(encoder_out, (nn_input.shape[0], self.bneck, 1, 1))
        return self.decoder(encoder_out)


class ConvDiscriminator(nn.Module):
    """
    Discriminator network
    """

    def __init__(self, img_size):
        super().__init__()
        width = 9680
        self.stack = nn.Sequential(
            nn.Conv2d(6, 5, (5, 5), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(5, 10, (5, 5), stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(10, 20, (5, 5), stride=2),
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.Linear(width, 1),
            nn.Sigmoid()
        )

    def forward(self, nn_input):
        return self.stack(nn_input)


# get summary of dimensions
net = ConvDiscriminator((200, 200))
torchsummary.summary(net, (6, 200, 200), 64)
