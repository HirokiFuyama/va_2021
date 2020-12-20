from dataclasses import dataclass
import time
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from src.vae.vae import VAE, ImgDataset, ImageTransform


@dataclass
class Config:
    lr: float = 1e-3
    beta1:float = 0.9
    beta2:float = 0.9
    input_dim: int = 500
    num_epoch: int = 100


def loss_function(recon_x, x, mu, logvar, config=Config()):
    bce = F.binary_cross_entropy_with_logits(recon_x, x, x.view(-1, config.input_dim), reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


def train(model=VAE, condig=Config(), train_dataloader, eval_dataloader):

    # check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Use device：", device)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=condig.lr, betas=[condig.beta1, condig.beta2])

    iteration = 1
    for epoch in range(condig.num_epoch):
        t_epoch_start = time.time()

        print('-------------')
        print('Epoch {}/{}'.format(epoch, condig.num_epoch))
        print('-------------')
        print('（train）')

        # ---------------------------------------------------------------------------------

        model.train()
        train_epoch_loss = 0
        for images in train_dataloader:

            images = images.to(device)
            pred, mu, logvar = model(images)

            loss = loss_function(pred, images, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

        print('-------------')
        print('Train epoch loss:{:.4f}'.format(train_epoch_loss / train_dataloader.batch_size))

        # -------------------------------------------------------------------------------

        model.eval()
        eval_epoch_loss = 0
        for images in eval_dataloader:

            images = images.to(device)
            pred, mu, logvar = model(images)

            loss = loss_function(pred, images, mu, logvar)

            eval_epoch_loss += loss.item()


    return model


def process():

    image_list():
