import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, dims):
        super(Reshape, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.view(-1, self.dims[0], self.dims[1], self.dims[2])


def weights_init(m):
    """
    changing the weights to a notmal distribution with mean=0 and std=0.01
    """
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 0.01)


class lw_ae(nn.Module):
    """
    model of autoencoder. this model allows to train each layer separately while freezing the other layers
    """
    def __init__(self, dataset_size, dims=[500, 500, 2000, 10], dropout=0.2):
        super(lw_ae, self).__init__()
        self.encoders = []
        self.decoders = []
        self.dropout = dropout
        self.init_stddev = 0.01

        w = dataset_size[1]
        if len(dataset_size)>2:
            w2 = dataset_size[2]
            channels = dataset_size[3]
            self.encoders.append(nn.Sequential(
                nn.Dropout(self.dropout),
                Flatten(),
                nn.Linear(dims[0], dims[1]),
                nn.ReLU()
            ))
            self.decoders.append(nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(dims[1], dims[0]),
                Reshape([channels, w, w2])
                ))
        else:
            self.encoders.append(nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(dims[0], dims[1]),
                nn.ReLU()
            ))
            self.decoders.append(nn.Sequential(
                nn.Dropout(self.dropout),  
                nn.Linear(dims[1], dims[0]),
            ))
        if len(dims)>1:
            for i in range(1, len(dims)-1):
                if i == (len(dims)-2):
                    self.encoders.append(nn.Sequential(
                        nn.Dropout(self.dropout),
                        nn.Linear(dims[i], dims[i+1]),
                    ))
                else:
                    self.encoders.append(nn.Sequential(
                        nn.Dropout(self.dropout),
                        nn.Linear(dims[i], dims[i+1]),
                        nn.ReLU()
                    ))
                self.decoders.append(nn.Sequential(
                    nn.Dropout(self.dropout),
                    nn.Linear(dims[i+1], dims[i]),
                    nn.ReLU()
                ))

        self.n_layers = len(self.encoders)
        for i in range(0, self.n_layers):
            self.encoders[i].apply(weights_init)
            self.decoders[i].apply(weights_init)

        # the complete autoencoder
        self.encoder = nn.Sequential(*self.encoders)
        decoder = list(reversed(self.decoders))
        self.decoder = nn.Sequential(*decoder)
        #frozen layers and trainable layers
        self.frozen_encoder = []
        self.frozen_decoder = []
        self.training_encoder = None
        self.training_decoder = None
        self.trainable_params = None
        self.layerwise_train = True


    def add_layer(self):
        """
        changing the current trainable layer to a frozen layer and the next layer to a trainable layer until
        there are no more layers to train
        """
        if self.training_encoder:
            # freezing the current trainable layer
            self.training_encoder.requires_grad = False
            self.frozen_encoder.append(self.training_encoder)
            self.training_decoder.requires_grad = False
            self.frozen_decoder.append(self.training_decoder)
        try:
            # adding a new layer to train
            self.training_encoder = self.encoders.pop(0)
            self.training_decoder = self.decoders.pop(0)
            self.trainable_params = [{'params': self.training_encoder.parameters()},
                                     {'params': self.training_decoder.parameters()}
                                     ]
        except:
            print('No more standby layers!')
            # update the complete autoencoder to include the trained layers
            self.encoder = nn.Sequential(*self.frozen_encoder)
            self.decoder = nn.Sequential(*list(reversed(self.frozen_decoder)))
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True
            for i in range(0, len(list(self.encoder))):
                self.encoder[i][0].train(False)
                self.decoder[i][0].train(False)

    def forward(self, x):
        if self.layerwise_train:
            # forward pass when training one layer at a time
            for e in self.frozen_encoder:
                x = e(x)
            encoded = self.training_encoder(x)
            decoded = self.training_decoder(encoded)
            for d in list(reversed(self.frozen_decoder)):
                decoded = d(decoded)
        else:
            #forward pass when training the entire autoencoder
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
        return encoded, decoded
