import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Network(pl.LightningModule):
    def __init__(self, seed, input_size, h_sizes,lr, dropout):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.lr = lr
        self.net = torch.nn.Sequential()
        self.dropout = dropout
        for k in range(len(h_sizes)):
            self.net.add_module(str('hidden_'+str(k)),nn.Linear(input_size, h_sizes[k]))
            self.net.add_module(str('dropout_'+str(k)),torch.nn.Dropout(self.dropout))
            self.net.add_module(str('tanh_'+str(k)),nn.Tanh())
            input_size = h_sizes[k]
        self.net.add_module('output',nn.Linear(h_sizes[-1], 1))
        self.net.add_module('sigmoid',nn.Sigmoid())
    
    def forward(self, x):
        layers = self.net(x)
        return layers
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x,y = train_batch
        logits = self.forward(x)
        loss = F.mse_loss(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.mse_loss(logits, y)
        self.log('val_loss', loss, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)



