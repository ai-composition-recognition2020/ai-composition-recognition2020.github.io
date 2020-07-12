import torch
import random
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class rp(nn.Module):
    def __init__(self, batch_size):
        super(rp, self).__init__()
        self.batch_size = batch_size

    def forward(self, x):
        # 10*128*128
        x = x.view(self.batch_size*128, -1)
        return x


class AE(nn.Module):
    def __init__(self, in_dim, hidden_dim, batch_size, test=False):
        super(AE, self).__init__()

        if test:
            self.batch_size = 1
        else:
            self.batch_size = batch_size

        self.encoder = nn.Sequential(
            # input: 10*128
            nn.Embedding(num_embeddings=128, embedding_dim=128), # 1* 128 * 128
            nn.Linear(in_dim, in_dim),# 1* 128 * 128
            #  nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),# 1* 128 * 128
            #  nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),# 1* 128 * 128
            #  nn.BatchNorm1d(),
            nn.ReLU(),
            # 128 * 32
            nn.Linear(in_dim, hidden_dim),# 1* 128 * 32
            #  nn.BatchNorm1d(),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            # input: 1*128*32
            nn.Linear(hidden_dim, in_dim), # 1* 128 * 128
            #  nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim), # 1* 128 * 128
            #  nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim), # 1* 128 * 128
            #  nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim), # 1* 128 * 128
            #  nn.BatchNorm1d(),
            nn.ReLU(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        hidden = self.encoder(x)
        out = self.decoder(hidden) # 1*128
        out = out.view(self.batch_size*128, -1)
        out = F.softmax(out, dim=1) # 1*128

        return out
