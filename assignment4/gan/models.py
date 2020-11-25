import torch
import torch.nn as nn
from gan.spectral_normalization import SpectralNorm

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        #Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.conv1 = SpectralNorm(nn.Conv2d(3, 128, 4, stride=2, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(128, 256, 4, stride=2, padding=1))
        self.bn1 = nn.BatchNorm2d(256)
        self.conv3 = SpectralNorm(nn.Conv2d(256, 512, 4, stride=2, padding=1))
        self.bn2 = nn.BatchNorm2d(512)
        self.conv4 = SpectralNorm(nn.Conv2d(512, 1024, 4, stride=2, padding=1))
        self.bn3 = nn.BatchNorm2d(1024)
        self.conv5 = SpectralNorm(nn.Conv2d(1024, 1, 4, stride=1, padding=0))
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv2d(3, 128, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1, 4, stride=1, padding=0)
        self.leakyrelu = nn.LeakyReLU(0.2)

        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.bn1(x)
        x = self.leakyrelu(self.conv3(x))
        x = self.bn2(x)
        x = self.leakyrelu(self.conv4(x))
        x = self.bn3(x)
        x = self.conv5(x)
        # x = self.leakyrelu(x)
        batch_size = x.shape[0]
        x = x.view(batch_size,1)
        ##########       END      ##########

        return x


class Generator(nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.model = nn.Sequential(
            nn.ConvTranspose2d(self.noise_dim, 1024, 4, stride = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, stride = 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride = 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride = 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, stride = 2, padding=1),
            nn.Tanh()
        )

        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        x = self.model(x)
        ##########       END      ##########

        return x
