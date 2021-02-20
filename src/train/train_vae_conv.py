import sys
import glob
import time
import torch
import torchvision
import numpy as np
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from dataclasses import dataclass
# import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.preprocess.image_loader import ImgDataset, ImageTransform
from src.vae.vae_conv import VAE


@dataclass
class Config:
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.9
    input_dim: int = 128
    num_epoch: int = 1000
    num_stopping: int = 5
    batch_size: int = 64
    z_dim: int = 32
    save_path: str = '../../model/vae/vae.pt'
    log_path: str = '../../log/vae/vae_lr_1e-4'



def loss_function(recon_x, x, mu, logvar, config):
    bce = F.binary_cross_entropy_with_logits(recon_x.view(-1, config.input_dim), x.view(-1, config.input_dim), reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


def train(train_dataloader, eval_dataloader, model, config):

    # check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Use device：", device)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=[config.beta1, config.beta2])

    writer = SummaryWriter(log_dir=config.log_path)
    
    models = []
    eval_loss = []
    generated_images = []
    for epoch in range(config.num_epoch):
        t_epoch_start = time.time()

        print('-------------')
        print('Epoch {}/{}'.format(epoch, config.num_epoch))
        print('-------------')

        # ---------------------------------------------------------------------------------

        model.train()
        train_epoch_loss = 0
        n_t = 0
        for images in train_dataloader:

            images = images.to(device)

            # For liner vae
            # images = images.view(-1, config.input_dim)

            pred, mu, logvar = model(images)
            
            loss = loss_function(pred, images, mu, logvar, config)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
            n_t += 1

        # print('Train epoch loss:{:.4f}'.format(train_epoch_loss / train_dataloader.batch_size))
        print('Train epoch loss:{:.4f}'.format(train_epoch_loss / n_t))

        # -------------------------------------------------------------------------------

        model.eval()
        eval_epoch_loss = 0
        n_e = 0
        for images in eval_dataloader:

            images = images.to(device)

            # For liner vae
            # images = images.view(-1, config.input_dim)

            pred, mu, logvar = model(images)

            loss = loss_function(pred, images, mu, logvar, config)

            eval_epoch_loss += loss.item()
            n_e += 1

        models.append(model)

        # Early stopping -------------------------------------------------------

        eval_loss.append(eval_epoch_loss/n_e)

        # To tensor board
        writer.add_scalar('Train/loss', train_epoch_loss / n_t, epoch)
        writer.add_scalar('Eval/loss', eval_epoch_loss / n_e, epoch)

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
        if epoch % 10 == 0:
            _generated = torchvision.utils.make_grid(pred[:10], nrow=5)
            _true = torchvision.utils.make_grid(images[:10], nrow=5)
            writer.add_image('Eval/generated', _generated, epoch)
            writer.add_image('Eval/true', _true, epoch)


    return models[low_index + 1], generated_images


def process(train_dir_path, eval_dir_path, config):

    # params of normalization
    _mean = 0.5
    _std = 0.5

    # read file path
    train_path_list = glob.glob(train_dir_path)
    eval_path_list = glob.glob(eval_dir_path)

    if train_path_list == [] or eval_path_list == []:
        print('FileNotFoundError: No such file or directory: ', file=sys.stderr)
        sys.exit(1)

    # mk dataloader
    train_dataset = ImgDataset(file_list=train_path_list, transform=ImageTransform(_mean, _std))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    eval_dataset = ImgDataset(file_list=eval_path_list, transform=ImageTransform(_mean, _std))
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=True)

    vae = VAE(config.input_dim, config.z_dim)

    # train model
    model, generated = train(train_dataloader, eval_dataloader, vae, config)

    # save model
    torch.save(model.state_dict(), config.save_path)

    return generated


if __name__ == '__main__':
    t_dir_path = rf'..\..\figure\spectrogram_png\train\*.png'
    e_dir_path = rf'..\..\figure\spectrogram_png\test\*.png'
    process(t_dir_path, e_dir_path, Config())
