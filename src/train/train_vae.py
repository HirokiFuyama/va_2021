import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F

from src.preprocess.image_loader import ImgDataset, ImageTransform
from src.vae.vae import VAE


# @dataclass
# class Config:
#     lr: float = 1e-5
#     beta1: float = 0.9
#     beta2: float = 0.9
#     input_dim: int = 16384
#     num_epoch: int = 100
#     num_stopping: int = 50
#     batch_size: int = 256
#     z_dim: int = 50
#     save_path: str = '../../model/vae.pt'


def loss_function(recon_x, x, mu, logvar, config):
    bce = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, config.input_dim), reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


def train(train_dataloader, eval_dataloader, model, config):

    # check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Use device：", device)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=[config.beta1, config.beta2])

    models = []
    eval_loss = []
    for epoch in range(config.num_epoch):
        t_epoch_start = time.time()

        print('-------------')
        print('Epoch {}/{}'.format(epoch, config.num_epoch))
        print('-------------')

        # ---------------------------------------------------------------------------------

        model.train()
        train_epoch_loss = 0
        for images in train_dataloader:

            images = images.to(device)

            # For liner vae
            images = images.view(-1, config.input_dim)

            pred, mu, logvar = model(images)

            loss = loss_function(pred, images, mu, logvar, config)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

        print('Train epoch loss:{:.4f}'.format(train_epoch_loss / train_dataloader.batch_size))

        # -------------------------------------------------------------------------------

        model.eval()
        eval_epoch_loss = 0
        n_e = 0
        for images in eval_dataloader:

            images = images.to(device)

            # For liner vae
            images = images.view(-1, config.input_dim)

            pred, mu, logvar = model(images)

            loss = loss_function(pred, images, mu, logvar, config)

            eval_epoch_loss += loss.item()
            n_e += 1

        models.append(model)

        # Early stopping -------------------------------------------------------

        eval_loss.append(eval_epoch_loss/n_e)

        if epoch >= config.num_stopping:
            if epoch == config.num_stopping:
                low_loss = np.min(eval_loss)
                low_index = np.argmin(eval_loss)
                if low_index == 0:
                    print('-------------------------------------------------------------------------------------------')
                    print("Early stopping")
                    print('Best Iteration:{}'.format(low_index+1))
                    print('Best evaluation loss:{}'.format(low_loss))
                    break

            elif epoch == low_index + config.num_stopping:
                low_loss_new = np.min(eval_loss[low_index:])
                low_index_new = np.argmin(eval_loss[low_index:])+low_index

                if low_loss <= low_loss_new:
                    print('-------------------------------------------------------------------------------------------')
                    print("Early stopping")
                    print('Best Iteration:{}'.format(low_index + 1))
                    print('Best evaluation loss:{}'.format(low_loss))
                    break
                else:
                    low_loss = low_loss_new
                    low_index = low_index_new
        else:
            pass

        t_epoch_finish = time.time()
        print('Eval_Epoch_Loss:{:.4f}'.format(eval_epoch_loss / n_e))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

        # check generated image ---------------------------------------------------
        if epoch % 20 == 0:
            x = pred.to('cpu').detach().numpy().copy()
            x = x[0].reshape(128, 128)
            plt.imshow(x)
            plt.show()

    return models[low_index + 1]


def process(train_dir_path, eval_dir_path, config):

    # params of normalization
    _mean = 0
    _std = 1

    # read file path
    train_path_list = glob.glob(train_dir_path)
    eval_path_list = glob.glob(eval_dir_path)

    # mk dataloader
    train_dataset = ImgDataset(file_list=train_path_list, transform=ImageTransform(_mean, _std))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    eval_dataset = ImgDataset(file_list=eval_path_list, transform=ImageTransform(_mean, _std))
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=True)

    vae = VAE(config.input_dim, config.z_dim)

    # train model
    model = train(train_dataloader, eval_dataloader, vae, config)

    # save model
    torch.save(model.state_dict(), config.save_path)

    # return model


# if __name__ == '__main__':
#     t_dir_path = ''
#     e_dir_path = ''
#     process(t_dir_path, e_dir_path)