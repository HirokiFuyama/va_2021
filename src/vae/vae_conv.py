import torch
from torch import nn

class VAE(nn.Module):

    def __init__(self, image_size, z_dim):
        super(VAE, self).__init__()
        self.input_dim = image_size

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, image_size, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(image_size, image_size*2, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.25)
            )

        self.layer3 = nn.Sequential(
            nn.Conv2d(image_size, image_size*4, kernel_size=4,
                      stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(0.25)
        )

        self.layer4 = nn.Linear(image_size*4, z_dim)  # mu
        self.layer5 = nn.Linear(image_size*4, z_dim)  # logvar

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, image_size*4,
                               kernel_size=4, stride=1),
            nn.BatchNorm2d(image_size*4),
            nn.ReLU(inplace=True))

        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(image_size*2, image_size*2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size*2),
            nn.ReLU(inplace=True))

        self.layer8 = nn.Linear(image_size*2, image_size)

    def encode(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        mu = self.layer4(x)
        logvar = self.layer5(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        #  randn_like : Gaussian Arg:Tensor
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = self.layer6(z)
        x = self.layer7(x)
        out = self.layer8(x)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        return z,  mu, logvar
