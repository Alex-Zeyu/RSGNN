import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear
import torch.nn.functional as F
class SignedConv(nn.Module):
    def __init__(self, in_channels, out_channels, first_aggr: bool, cuda = False):
        super(SignedConv,self).__init__()
        self.adj = None
        self.pos_mask = None
        self.neg_mask = None
        self.pos_adj = None
        self.neg_adj = None
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.first_aggr = first_aggr

        if first_aggr:
            self.lin_pos = Linear(2 * self.in_channels, self.out_channels // 2)
            self.lin_neg = Linear(2 * self.in_channels, self.out_channels // 2)
        else:
            self.lin_pos = Linear(3 * self.in_channels //2, self.out_channels // 2)
            self.lin_neg = Linear(3 * self.in_channels //2, self.out_channels // 2)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_pos.reset_parameters()
        self.lin_neg.reset_parameters()

    def forward(self, x, adj):
        self.adj = adj
        self.pos_mask = (self.adj >= 0).int()
        self.neg_mask = (self.adj <= 0).int()
        self.pos_adj = self.adj * self.pos_mask
        self.neg_adj = self.adj * self.neg_mask * -1

        # self.pos_adj = self._normalized(self.pos_adj)
        # self.neg_adj = self._normalized(self.neg_adj)
        self.pos_adj = F.normalize(self.pos_adj, p=1)
        self.neg_adj = F.normalize(self.neg_adj, p=1)

        if self.first_aggr:
            out_pos = torch.cat([torch.mm(self.pos_adj, x), x], dim = 1)
            out_pos = self.lin_pos(out_pos)

            out_neg = torch.cat([torch.mm(self.neg_adj, x), x], dim = 1)
            out_neg = self.lin_neg(out_neg)

            return torch.cat([out_pos, out_neg], dim = 1)
        else:
            x_dim = x.shape[1]
            x_pos = x[:, 0:x_dim//2]
            x_neg = x[:, x_dim//2:]

            out_pos = torch.cat([torch.mm(self.pos_adj, x_pos),
                                 torch.mm(self.neg_adj, x_neg),
                                 x_pos], dim =1)
            out_pos = self.lin_pos(out_pos)

            out_neg = torch.cat([torch.mm(self.pos_adj, x_neg),
                                 torch.mm(self.neg_adj, x_pos),
                                 x_neg], dim =1)
            out_neg = self.lin_neg(out_neg)

            return torch.cat([out_pos, out_neg], dim =1)


    def _normalized(self, adj):
        deg = adj.sum(dim=1,keepdim= True)
        adj = adj / deg
        adj.nan_to_num_(nan=0.0)
        return adj








