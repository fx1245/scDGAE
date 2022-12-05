import torch
import torch.nn.functional as F
from gae.layer import GraphAttentionLayer,MultiHeadAttentionLayer


class GATModelVAE(torch.nn.Module):
    def __init__(self, input_dim, h_dim, z_dim, heads):
        super(GATModelVAE, self).__init__()

        #### Encoder ####
        self.heads = heads
        self.encode_conv1 = MultiHeadAttentionLayer( in_features=input_dim, out_features=h_dim,
                                  heads=3, concat=False)
        self.encode_bn1 = torch.nn.BatchNorm1d(h_dim)

        self.encode_conv2 = MultiHeadAttentionLayer( in_features=h_dim, out_features=z_dim,
                                  heads=3, concat=False)
        self.encode_bn2 = torch.nn.BatchNorm1d(z_dim)

        #### Decoder ####
        self.decode_linear1 = torch.nn.Linear(z_dim, h_dim)
        self.decode_bn1 = torch.nn.BatchNorm1d(h_dim)

        self.decode_linear2 = torch.nn.Linear(h_dim, input_dim)

    def encode(self, x, edge_index):
        # hidden1 = self.encode_conv1(x,edge_index)
        # return self.encode_conv2(hidden1,edge_index)
        x = F.relu(self.encode_bn1(self.encode_conv1(x, edge_index)))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.encode_bn2(self.encode_conv2(x, edge_index)))
        x = F.dropout(x, p=0.5, training=self.training)

        return x

    def decode(self, x):
        x = F.relu(self.decode_bn1(self.decode_linear1(x)))
        x = F.relu(self.decode_linear2(x))

        return x

    # def reparameterize(self, mu, logvar):
    #     if self.training:
    #         std = torch.exp(logvar)
    #         eps = torch.randn_like(std)
    #         return eps.mul(std).add_(mu)
    #     else:
    #         return mu

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x = self.decode(z)
        # x = x * size_factors
        return x


    # def forward(self, x, edge_index,):
    #     mu, logvar = self.encode(x, edge_index)
    #     z = self.reparameterize(mu, logvar)
    #     return z, mu, logvar


class InnerProductDecoder(torch.nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        edge_index = self.act(torch.mm(z, z.t()))
        return edge_index


# class GCNModelAE(torch.nn.Module):
#     def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
#         super(GCNModelAE, self).__init__()
#         self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
#         self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
#         self.dc = InnerProductDecoder(dropout, act=lambda x: x)
#
#     def encode(self, x, adj):
#         hidden1 = self.gc1(x, adj)
#         return self.gc2(hidden1, adj)
#
#     def forward(self, x, adj, encode=False):
#         z = self.encode(x, adj)
#         return z, z, None
