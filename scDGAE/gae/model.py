import torch
import torch.nn as nn
import torch.nn.functional as F
from gae.layer import layer
import theano.tensor as T
from layer import *
from LossCalculation import Dual_Loss
import layer as la


class GATModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout=0., layerType='GATConv', heads=5 ):
        super(GATModelVAE, self).__init__()
        self.gc1 = layer(layerType, dropout=dropout, in_channels=input_feat_dim, out_channels=hidden_dim1, heads=heads, act=F.relu, concat=False)
        self.gc2 = layer(layerType, dropout=dropout, in_channels=hidden_dim1, out_channels=hidden_dim2, heads=heads, act=lambda x:x, concat=False)
        self.gc3 = layer(layerType, dropout=dropout, in_channels=hidden_dim1, out_channels=hidden_dim2, heads=heads,  act=lambda x:x, concat=False)
        self.dc = InnerProductDecoder(dropout,  act=lambda x:x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act


    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class DGCN(object):
    def __init__(self, rng, input, layer_sizes, diffusion, ppmi,
                 dropout_rate=0.3, activation=None, nell_dataset=False):

        self.a_layers = []
        self.ppmi_layers = []
        self.l1 = 0.
        self.l2 = 0.
        self.loss = None
        self.input = input  #### X

        # define the dual NN sharing the same weights Ws
        next_a_layer_input = input
        next_ppmi_layer_input = input

        for s in layer_sizes:
            _hiddenLayer_a = HiddenDenseLayer(
                rng=rng,
                input=next_a_layer_input,
                n_in=s[0],
                n_out=s[1],
                diffusion=diffusion)
            self.a_layers.append(_hiddenLayer_a)

            _hiddenLayer_ppmi = HiddenDenseLayer(
                rng=rng,
                input=next_ppmi_layer_input,
                n_in=s[0],
                n_out=s[1],
                diffusion=ppmi,
                W=_hiddenLayer_a.W)  #### share the same weight matrix W
            self.ppmi_layers.append(_hiddenLayer_ppmi)
            # drop out
            _layer_output_a = _hiddenLayer_a.output
            _layer_output_ppmi = _hiddenLayer_ppmi.output
            next_a_layer_input = la._dropout_from_layer(rng, _layer_output_a, dropout_rate)
            next_ppmi_layer_input = la._dropout_from_layer(rng, _layer_output_ppmi, dropout_rate)

        # record all the params to do training
        self.params = [param for layer in self.a_layers for param in layer.params]

        # define the NN output
        self.a_output = T.nnet.softmax(self.a_layers[-1].output)
        self.ppmi_output = T.nnet.softmax(self.ppmi_layers[-1].output)

        # define the regulizer
        for _W in self.params:
            self.l2 += (_W ** 2).sum() / 2.0
            self.l1 += abs(_W).sum()

        self.LossCal = Dual_Loss(self.a_output, self.ppmi_output)

        # define the supervised loss
        if nell_dataset:
            self.supervised_loss = self.LossCal.masked_cross_entropy
        else:
            self.supervised_loss = self.LossCal.masked_mean_square

        # define the unsupervised loss
        self.unsupervised_loss = self.LossCal.unsupervised_loss

        # define the test accuracy function
        self.acc = self.LossCal.acc

    # def forward(self, x, edge_index):
    #     z = self.encode(x, edge_index)
    #     x = self.decode(z)
    #    # x = x * size_factors
    #     return x
    #
    # def forward(self, x, edge_index):
    #     mu, logvar = self.encode(x, edge_index)
    #     z = self.reparameterize(mu, logvar)
    #     return z, mu, logvar
