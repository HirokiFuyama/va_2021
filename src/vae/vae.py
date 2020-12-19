import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import torch.utils.data as data
from torch.nn import functional as F


class VAE(nn.Module):

    def __init__(self, input_dim, z_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc21 = nn.Linear(256, z_dim)  # mu
        self.fc22 = nn.Linear(256, z_dim)  # logvar
        self.fc3 = nn.Linear(z_dim, 256)
        self.fc4 = nn.Linear(256, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        #  randn_like : Gaussian Arg:Tensor
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        return z,  mu, logvar


class ImageTransform:
    """
    This class is preprocessing image.
    """
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)


class ImgDataset(data.Dataset):

    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        """
        Convert image to Tensor data type.
        """
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        return img_transformed
