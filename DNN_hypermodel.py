import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import optuna
from optuna.trial import TrialState

class Network(pl.LightningModule):
    def __init__(self, seed, input_size, n_layers, dropouts, out_features,lr):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.lr = lr
        self.net = torch.nn.Sequential()
        for k in range(n_layers):
            out_feature = out_features[k]
            self.net.add_module(str('hidden_'+str(k)),nn.Linear(input_size, out_feature))
            self.net.add_module(str('tanh_'+str(k)),nn.Tanh())
            self.net.add_module(str('dropout_'+str(k)),torch.nn.Dropout(dropouts[k]))
            input_size = out_feature
        self.net.add_module('output',nn.Linear(out_feature, 1))
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
        #self.log('val_loss', loss, prog_bar=True)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)



