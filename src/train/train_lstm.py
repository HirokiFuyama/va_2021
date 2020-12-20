import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from dataclasses import dataclass
import time
import itertools
# from scipy import signal


df = pd.read_csv()


@dataclass
class Config:
    # lstm-----------------------------------------------------------------------------------
    lr: float = 0.000006
    beta1: float = 0.8
    beta2: float = 0.7
    num_batch: int = 64
    num_epoch: int = 5000
    num_stopping: int = 10
    # save_path: str = ''
    # highpass ------------------------------------------------------------------------------
    # gpass_h: int = 1  # dB
    # gstop_h: int = 40  # dB
    # fp_h: int = 0.8  # Hz
    # fp_s_h: int = 0.1  # Hz
    # data ----------------------------------------------------------------------------------
    window = 250  # number of sample


# def highpass(data, config=Config(), fs=50):
#     gpass = config.gpass_h
#     gstop = config.gstop_h
#     fp = config.fp_h
#     fp_s = config.fp_s_h
#     # ----------------------------
#     Wp_d = fp / (fs / 2)
#     Ws_d = fp_s / (fs / 2)
#     N_d, Wn_d = signal.buttord(Wp_d, Ws_d, gpass, gstop)
#     b_d, a_d = signal.butter(N_d, Wn_d, "high")
#     return signal.filtfilt(b_d, a_d, data)


def onehot(y):
    y_df = pd.DataFrame(y).astype(str)
    y_df = pd.get_dummies(y_df, columns=[0])
    return y_df.values


def data_split(df, window=Config().window):
    x = df[['axl_x', 'axl_y', 'axl_z']].values.reshape(-1, window, 3)
    y = df['major_activity'].values[::window]

    # high pass
    # for length in range(len(x)):
    #     for features in range(len(x[length].T)):
    #         x[length].T[features] = highpass(x[length].T[features])

    # onehot encording
    y = onehot(y)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    for train_index, test_index in sss.split(x_train, y_train):
        x_train, x_eval = x_train[train_index], x_train[test_index]
        y_train, y_eval = y_train[train_index], y_train[test_index]

    return x_train, x_eval, x_test, y_train, y_eval, y_test


class Lstm(nn.Module):

    def __init__(self, hidden=256, layer=1, feature=3):
        super(Lstm, self).__init__()
        self.hidden = hidden

        self.layer1 = nn.LSTM(input_size=feature,
                              hidden_size=hidden,
                              num_layers=layer,
                              batch_first=True,
                              bidirectional=True)

        # self.layer2 = nn.Sequential(
        #     nn.Linear(in_features=hidden,
        #               out_features=128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=hidden*2,
                      out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU())

        self.layer3 = nn.Sequential(
            nn.Linear(in_features=128,
                      out_features=64),
            nn.Dropout(0.3),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=64,
                      # out_features=3),
                      out_features=3),
            nn.Softmax(dim=1))

    def forward(self, x):
        # out, _ = self.layer1(x)
        # out = self.layer2(out[:, -1, :])

        # For BLSTM
        _, out = self.layer1(x)
        out = torch.cat([out[0][0], out[0][1]], dim=1)
        out = self.layer2(out)

        out = self.layer3(out)
        out = self.layer4(out)
        return out


class LstmDataset(data.Dataset):

    def __init__(self, array, label, transform=None):
        self.array = array
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        x = self.array[index]
        y = self.label[index]
        return x, y


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 1.0)
        nn.init.constant_(m.bias.data, 0)

    elif classname.find('LSTM') != -1:
        # The number of weights varies with the number of layers.
        nn.init.normal_(m.weight_ih_l0, 0, 1.0)
        nn.init.normal_(m.weight_hh_l0, 1.0, 1.0)
        nn.init.constant_(m.bias_ih_l0, 0)
        nn.init.constant_(m.bias_hh_l0, 0)


def train_model(model, train_dataloader, eval_dataloader, config):

    # Check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Use device：", device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), config.lr, [config.beta1, config.beta2])
    # criterion = nn.CrossEntropyLoss()  # must not one hot
    criterion = nn.BCELoss().to(device)  # must one hot

    # To GPU and train mode
    model.to(device)

    # Accelerator
    torch.backends.cudnn.benchmark = True

    # num_train_imgs = len(dataloader.dataset)
    eval_loss = []
    # iteration = 1
    models = []
    for epoch in range(config.num_epoch):
        t_epoch_start = time.time()

        print('---------------------------------------------')
        print('Epoch {}/{}'.format(epoch, config.num_epoch))
        print('---------------------------------------------')
        # --------------------------------------------------------------------------------------------------------------

        model.train()
        t_epoch_loss = 0.0
        n_t = 0
        for data, label in train_dataloader:

            data = data.to(device)
            # label = label.long()  # for nn.CrossEntropyLoss
            label = label.float()  # for nn.BCELoss
            label = label.to(device)

            pred = model(data.float())
            # pred = pred.T

            # loss
            t_loss = criterion(pred, label)

            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()

            # save loss
            t_epoch_loss += t_loss.item()

            # iteration += 1
            n_t += 1

        # print loss
        # t_epoch_finish = time.time()
        # print('--------------------')
        print('Trian_Epoch_Loss:{:.4f}'.format(t_epoch_loss / n_t))

        # --------------------------------------------------------------------------------------------------------------

        model.eval()
        e_epoch_loss = 0.0
        n_e = 0
        for data, label in eval_dataloader:

            data = data.to(device)
            # label = label.long()  # for nn.CrossEntropyLoss
            label = label.float()  # for nn.BCELoss
            label = label.to(device)

            pred = model(data.float())

            # loss
            e_loss = criterion(pred, label)

            # save loss
            e_epoch_loss += e_loss.item()

            n_e += 1
        models.append(model)

        # Early stopping -------------------------------------------------------
        eval_loss.append(e_epoch_loss/n_e)
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
        print('Eval_Epoch_Loss:{:.4f}'.format(e_epoch_loss / n_e))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

    return models[low_index + 1]


def test_model(model, test_loader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred, y = [], []
    for data, label in test_loader:
        data = data.to(device)
        label = label.float()  # for nn.BCELoss

        pred.append(model(data.float()).argmax(axis=1).to('cpu').detach().numpy().copy())
        y.append(label.argmax(axis=1).to('cpu').detach().numpy().copy())

    return list(itertools.chain.from_iterable(pred)), list(itertools.chain.from_iterable(y))


def run(df, model, config):

    start = time.time()
    # split data--------------------------------------------------------------------------------------------------------
    x_train, x_eval, x_test, y_train, y_eval, y_test = data_split(df)
    train_dataloader = torch.utils.data.DataLoader(
        LstmDataset(array=x_train, label=y_train), batch_size=config.num_batch, shuffle=False
    )

    eval_dataloader = torch.utils.data.DataLoader(
        LstmDataset(array=x_eval, label=y_eval), batch_size=config.num_batch, shuffle=False
    )

    test_dataloader = torch.utils.data.DataLoader(
        LstmDataset(array=x_test, label=y_test), batch_size=config.num_batch, shuffle=False
    )

    # train model-------------------------------------------------------------------------------------------------------
    model.apply(weights_init)
    model = train_model(Lstm(), train_dataloader, eval_dataloader, config)
    torch.save(model.state_dict(), config.save_path)

    # test model--------------------------------------------------------------------------------------------------------
    pred, label = test_model(model, test_dataloader)
    print(confusion_matrix(pred, label))
    print(classification_report(pred, label))

    end = time.time()
    print('Run all time:{}'.format(end-start))

    # return pred, label


if __name__ == '__main()':
    run(df, Lstm(), Config())