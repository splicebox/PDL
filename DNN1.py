import os, sys, math, pickle, random
from collections import OrderedDict
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.nn.init as init

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multinomial import Multinomial

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser


class DSCNN(pl.LightningModule):
    def weight_init(self, size):
        cases = [torch.tensor([2, -2, -2, -2]), torch.tensor([-2, 2, -2, -2]),
                torch.tensor([-2, -2, 2, -2]), torch.tensor([-2, -2, -2, 2])]
        weight = torch.zeros([size, 1, 4, 18]).uniform_(-0.1, 0.1)
        # hexamer (6) motif
        for i in range(size // 2):
            for k in range(6, 12):
                l = np.random.randint(0, 4)
                weight[i, 0, :, k] = cases[l]

        # octamer (8) motif
        for i in range(size // 2, size):
            for k in range(5, 13):
                l = np.random.randint(0, 4)
                weight[i, 0, :, k] = cases[l]
        return weight

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        # self.load_data()

        rbp_channels = 1024
        self.rbp_cnn_layer = nn.Sequential(
            nn.Conv2d(1, rbp_channels, kernel_size=(4, 18), stride=1, bias=False),
            nn.BatchNorm2d(rbp_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
        )
        self.rbp_cnn_layer[0].weight = torch.nn.Parameter(self.weight_init(rbp_channels))

        embed_size = 256
        self.embedding_layer = nn.Sequential(
            nn.Conv1d(rbp_channels, embed_size, kernel_size=10, stride=2, bias=False),
            nn.BatchNorm1d(embed_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_size, embed_size, kernel_size=10, stride=2, bias=False),
            nn.BatchNorm1d(embed_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_size, embed_size, kernel_size=10, stride=2, bias=False),
            nn.BatchNorm1d(embed_size),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size, bias=False),
            nn.BatchNorm1d(embed_size),
            nn.ReLU(inplace=True),
            # nn.Linear(embed_size, embed_size),
            # nn.BatchNorm1d(embed_size),
            # nn.ReLU(inplace=True),
            # nn.Linear(embed_size, self.hparams.dims, bias=False),
            nn.Linear(embed_size, self.hparams.dims),
            # nn.BatchNorm1d(self.hparams.dims),
            # nn.Threshold(1e-5, 1e-5, inplace=True),
        )
        # self.dropout = nn.Dropout(0.1)
        # torch.autograd.set_detect_anomaly(True)

    def process_pwm_inputs(self, x):
        out = self.rbp_cnn_layer(x)
        out = out.squeeze(2)
        out = self.embedding_layer(out)  # shape: (batch, embed_size)
        out = torch.flatten(out, 1)
        return out

    def embedding_forward(self, x):
        # split the input into two regions
        size = x.shape[-1] // 2
        x1 = x[:, :, :, :size]
        x2 = x[:, :, :, size:]

        out1 = self.process_pwm_inputs(x1)
        out2 = self.process_pwm_inputs(x2)
        out = torch.cat([out1, out2], dim=-1)  # shape: (batch, embed_size * 2 + RBPs_size)
        # out = self.dropout(out)
        return out

    def forward(self, x, y, size):
        x = x.unsqueeze(1)  # shape: (batch, 1, 4, seq_length)
        embedded = self.embedding_forward(x)  # shape: (batch, embed_size)

        out = self.output_layer(embedded)  # shape: (batch, dims)
        out = out.view((-1, size, self.hparams.dims))

        loss = pre = 0
        for i, j in zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12, 24, 36, 40, 52, 64, 75, 87, 95, 102, 114, 120]):
            probs = Dirichlet(out[:, :, i].exp().clamp(1e-3, 1e3)).rsample((j - pre,))
            counts = y[:, :, pre:j].permute(2, 0, 1)
            loss += -Multinomial(probs=probs, validate_args=False).log_prob(counts).mean()
            pre = j
        loss = loss / self.hparams.dims

        return loss

    def training_step(self, batch, batch_idx):
        X, Y = batch
        size = X.size(1)
        X = X.view(-1, X.size(2), X.size(3))

        loss = self.forward(X, Y, size)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        size = X.size(1)
        X = X.view(-1, X.size(2), X.size(3))

        loss = self.forward(X, Y, size)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        X, Y = batch
        size = X.size(1)
        X = X.view(-1, X.size(2), X.size(3))

        loss = self.forward(X, Y, size)
        self.log('test_loss', loss, prog_bar=True, logger=True)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def configure_optimizers(self):
        my_list = ['rbp_cnn_layer.0.weight']
        params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list, self.named_parameters()))))
        base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in my_list, self.named_parameters()))))
        optimizer = Adam([{'params': base_params}, {'params': params, 'lr': 1e-2, 'weight_decay': 0}],
                         lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.2)

        return [optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=1, shuffle=True,
                        collate_fn=self.collate_fn, num_workers=4)

    def val_dataloader(self):
        # return DataLoader(self.test_set, batch_size=128)
        return DataLoader(self.test_set, batch_size=1,
                          collate_fn=self.collate_fn, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1,
                          collate_fn=self.collate_fn, num_workers=4)

    def collate_fn(self, batch):
        X, Y = [], []
        for x, y in batch[0]:
            X.append(x)
            Y.append(y)

        X = torch.tensor(X).float()
        Y = torch.tensor(Y).float()

        return X, Y

    def prepare_data(self):
        pkl_file = Path(self.hparams.data_dir) / 'data.pkl'  ##########################
        self.train_set, self.test_set = pickle.load(open(pkl_file, "rb"))


################################################################################
def main():
    parser = ArgumentParser()
    parser.add_argument('--dims', default=12, type=int)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--out-dir', default='.', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)

    hparams = parser.parse_args()


    model = DSCNN(hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=out_dir / 'DNN1',  ##########################
        filename='DSCNN-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    trainer = Trainer(callbacks=[checkpoint_callback],
                      # gradient_clip_val=1000,
                      progress_bar_refresh_rate=1,
                      max_epochs=200,
                      gpus=4,
                      distributed_backend='dp',
                      val_check_interval=0.1)

    result = trainer.fit(model)


if __name__ == "__main__":
  main()
