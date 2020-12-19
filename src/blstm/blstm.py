import torch
import torch.nn as nn


class Lstm(nn.Module):
    """
    3 class example
    """

    def __init__(self, hidden=256, layer=1, feature=3):
        super(Lstm, self).__init__()
        self.hidden = hidden

        self.layer1 = nn.LSTM(input_size=feature,
                              hidden_size=hidden,
                              num_layers=layer,
                              batch_first=True,
                              bidirectional=True)

        # for lstm
        # self.layer2 = nn.Sequential(
        #     nn.Linear(in_features=hidden,
        #               out_features=128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU())

        # for blstm
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
        # For Lstm
        # out, _ = self.layer1(x)
        # out = self.layer2(out[:, -1, :])

        # For BLSTM
        _, out = self.layer1(x)
        out = torch.cat([out[0][0], out[0][1]], dim=1)
        out = self.layer2(out)

        out = self.layer3(out)
        out = self.layer4(out)
        return out
