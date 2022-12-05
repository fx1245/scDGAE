import math
import numpy as np
import time
import os
import config

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_sparse import spmm   # product between dense matrix and sparse matrix
import torch_sparse as torchsp
from torch_scatter import scatter_add, scatter_max
import torch.sparse as sparse
import theano
import theano.tensor as T


def layer(layer_type, **kwargs):
    if layer_type == 'GCNConv':
        return GraphConvolution(in_features=kwargs['in_channels'], out_features=kwargs['out_channels'])
    elif layer_type == 'GATConv':
        return MultiHeadAttentionLayer(in_features=kwargs['in_channels'], out_features=kwargs['out_channels'],
                                       heads=kwargs['heads'], dropout=kwargs['dropout'], concat=kwargs['concat'])


class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0., alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'





class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, input_dim, out_dim, dropout=0., alpha=0.2, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = input_dim
        self.out_features = out_dim
        self.alpha = alpha
        self.concat = concat
        self.dropout = dropout
        self.W = nn.Parameter(torch.zeros(size=(input_dim, out_dim)))  # FxF'
        self.attn = nn.Parameter(torch.zeros(size=(1, 2 * out_dim)))  # 2F'
        nn.init.xavier_normal_(self.W, gain=1.414)
        nn.init.xavier_normal_(self.attn, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        '''
        :param x:   dense tensor. size: nodes*feature_dim
        :param adj:    parse tensor. size: nodes*nodes
        :return:  hidden features
        '''
        N = x.size()[0]  # 图中节点数
        edge = adj._indices()  # 稀疏矩阵的数据结构是indices,values，分别存放非0部分的索引和值，edge则是索引。edge是一个[2*NoneZero]的张量，NoneZero表示非零元素的个数
        if x.is_sparse:  # 判断特征是否为稀疏矩阵
            h = torch.sparse.mm(x, self.W)
        else:
            h = torch.mm(x, self.W)
        # Self-attention (because including self edges) on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()  # edge_h: 2*D x E
        values = self.attn.mm(edge_h).squeeze()  # 使用注意力参数对特征进行投射
        edge_e_a = self.leakyrelu(values)  # edge_e_a: E   attetion score for each edge，对应原论文中的添加leakyrelu操作
        # 由于torch_sparse 不存在softmax算子，所以得手动编写，首先是exp(each-max),得到分子
        edge_e = torch.exp(edge_e_a - torch.max(edge_e_a))
        # 使用稀疏矩阵和列单位向量的乘法来模拟row sum，就是N*N矩阵乘N*1的单位矩阵的到了N*1的矩阵，相当于对每一行的值求和
        e_rowsum = spmm(edge, edge_e, m=N, n=N,
                        matrix=torch.ones(size=(N, 1)))  # e_rowsum: N x 1，spmm是稀疏矩阵和非稀疏矩阵的乘法操作
        h_prime = spmm(edge, edge_e, n=N, m=N, matrix=h)  # 把注意力评分与每个节点对应的特征相乘
        h_prime = h_prime.div(
            e_rowsum + torch.Tensor([9e-15]))  # h_prime: N x out，div一看就是除，并且每一行的和要加一个9e-15防止除数为0
        # softmax结束
        if self.concat:
            # if this layer is not last layer
            return F.elu(h_prime)
        else:
            # if this layer is last layer
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, heads, concat=True):
        super(MultiHeadAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.attentions = [SpGraphAttentionLayer(in_features, out_features, concat=concat) for _ in range(heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = torch.cat([torch.unsqueeze(att(x, adj), 0) for att in self.attentions])
        x = torch.mean(x, dim=0)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class HiddenDenseLayer(object):
    def __init__(self, rng, input, n_in, n_out, diffusion, W=None,
                 activation=T.nnet.relu):

        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        self.W = W
        self.D = diffusion

        lin_output = theano.dot(theano.dot(diffusion, input), self.W)
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W]


# define a normal dense layer
class HiddenDenseLayer_normal(object):
    def __init__(self, rng, input, n_in, n_out, W=None,
                 activation=T.nnet.relu):

        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        self.W = W

        lin_output = theano.dot(input, self.W)
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W]


# Use to do the dropout process when training
def _dropout_from_layer(rng, layer, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(
        rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1 - p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output