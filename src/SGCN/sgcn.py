import sys
import torch
import torch.nn as nn
from .sgcn_conv import SignedConv
from torch_sparse import coalesce
import scipy.sparse
import torch.nn.functional as F
from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling)

class SignedGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 lamb=5, bias=True, op = 'cat'): # 64 64
        """
        The signed graph convolutional network model
        Args:
            in_channels:
            hidden_channels:
            num_layers:
            lamb:
            bias:
            op: edge representation operation ["cat", "mean", "add"]
        """
        super(SignedGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.lamb = lamb
        self.op = "cat"

        self.conv1 = SignedConv(in_channels, hidden_channels, first_aggr=True)
        self.convs = nn.ModuleList()
        for i in range(num_layers-1):
            self.convs.append(SignedConv(hidden_channels, hidden_channels,
                                         first_aggr=False))
        if op == "cat":
            self.lin = nn.Linear(2 * hidden_channels, 3)
        else:
            self.lin = nn.Linear(hidden_channels, 3)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, adj):
        z = torch.tanh(self.conv1(x, adj))
        for conv in self.convs:
            z = torch.tanh(conv(z, adj))
        return z

    def discriminate(self, z, edge_index):
        if self.op == "cat":
            edge_feature = torch.cat([z[edge_index[0]],z[edge_index[1]]], dim = 1)
        elif self.op == "mean":
            edge_feature = (z[edge_index[0]] + z[edge_index[1]]) /2
        elif self.op == "add":
            edge_feature = z[edge_index[0]] + z[edge_index[1]]

        logits = self.lin(edge_feature)
        return torch.log_softmax(logits, dim =1)

    def nll_loss(self, z, pos_edge_index, neg_edge_index, w_s = [1,1,1]):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim =1)
        non_edge_index = negative_sampling(edge_index, z.shape[0])

        nll_loss = 0
        nll_loss += F.nll_loss(
            self.discriminate(z, pos_edge_index),
            pos_edge_index.new_full((pos_edge_index.shape[1],),0)) * w_s[0]
        nll_loss += F.nll_loss(
            self.discriminate(z, neg_edge_index),
            neg_edge_index.new_full((neg_edge_index.shape[1], ),1))* w_s[1]
        nll_loss += F.nll_loss(
            self.discriminate(z, non_edge_index),
            non_edge_index.new_full((non_edge_index.shape[1], ),2))* w_s[2]

        # return nll_loss / (pos_edge_index.shape[1] +
        #                    neg_edge_index.shape[1]+
        #                    z.shape[0])
        return nll_loss /3

    def pos_embedding_loss(self, z, pos_edge_index):
        i, j , k = structured_negative_sampling(pos_edge_index, z.size(0))
        out =(z[i] - z[j]).pow(2).sum(dim =1) - (z[i] - z[k]).pow(2).sum(dim =1)
        return torch.clamp(out, min =0).mean()

    def neg_embedding_loss(self, z, neg_edge_index):
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))
        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def loss(self, z, pos_edge_index, neg_edge_index):
        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_pos = self.pos_embedding_loss(z, pos_edge_index)
        loss_neg = self.neg_embedding_loss(z, neg_edge_index)
        return nll_loss + self.lamb * (loss_pos + loss_neg)

    def test(self, z, pos_edge_index, neg_edge_index):
        """
        Evaluate node embedding 'z' by test edge index
        [pos_edge_index, neg_edge_index]
        Args:
            z: the node embeddings
            pos_edge_index: test positive edge indices
            neg_edge_index: test negative edge indices

        Returns: auc, f1

        """
        from sklearn.metrics import roc_auc_score, f1_score
        with torch.no_grad():
            pos_p = self.discriminate(z, pos_edge_index)[:,:2].max(dim=1)[1]
            neg_p = self.discriminate(z, neg_edge_index)[:,:2].max(dim=1)[1]

        pred = (1 - torch.cat([pos_p, neg_p])).cpu()
        y = torch.cat(
            [pred.new_ones((pos_p.size(0))),
             pred.new_zeros(neg_p.size(0))])
        pred, y = pred.numpy(), y.numpy()

        auc = roc_auc_score(y, pred)
        Binary_f1 = f1_score(y, pred, average='binary') if pred.sum() >0 else 0
        Micro_f1 = f1_score(y, pred, average='micro') if pred.sum() >0 else 0
        Macro_f1 = f1_score(y, pred, average='macro') if pred.sum() >0 else 0
        return auc, Binary_f1, Micro_f1, Macro_f1















