import os.path as osp
import time

# DIR_BASE = osp.dirname(osp.dirname(osp.abspath(__file__)))
# sys.path.append(DIR_BASE)

from src.utils import *
import torch
from sgcn import SignedGCN
from torch_geometric.utils import coalesce,to_undirected

from torch_geometric.datasets import BitcoinOTC
from torch_geometric.utils import to_dense_adj

name = 'BitcoinOTC-1'
path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', name)
dataset = BitcoinOTC(path, edge_window_size=1)

pos_edge_indices, neg_edge_indices = [], []


for data in dataset:
    pos_edge_indices.append(data.edge_index[:, data.edge_attr > 0])
    neg_edge_indices.append(data.edge_index[:, data.edge_attr < 0])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pos_edge_index = torch.cat(pos_edge_indices, dim=1).to(device)
neg_edge_index = torch.cat(neg_edge_indices, dim=1).to(device)

train_pos_edge_index, test_pos_edge_index = split_edges(pos_edge_index)
train_neg_edge_index, test_neg_edge_index = split_edges(neg_edge_index)

train_pos_edge_attr = torch.ones(train_pos_edge_index.shape[1])
train_neg_edge_attr = torch.ones(train_neg_edge_index.shape[1])* -1



train_edge = torch.cat([train_pos_edge_index, train_neg_edge_index], dim =1)
train_edge_attr = torch.cat([train_pos_edge_attr, train_neg_edge_attr]).to(device)

train_edge, train_edge_attr = coalesce(train_edge, train_edge_attr)

train_edge, train_edge_attr =  to_undirected(train_edge, train_edge_attr)

adj = to_dense_adj(edge_index=train_edge, edge_attr=train_edge_attr).squeeze().to(device)

x = create_spectral_features(train_pos_edge_index, train_neg_edge_index).to(device)
model = SignedGCN(64, 64, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)



def train():
    model.train()
    optimizer.zero_grad()
    z = model(x, adj)
    loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        z = model(x, adj)
    return model.test(z, test_pos_edge_index, test_neg_edge_index)


for epoch in range(200):
    t = time.time()
    loss = train()
    auc, f1 = test()
    t_cost = time.time() - t
    print(f'Epoch: {epoch:03d}, Time: {t_cost:.2f}s, Loss: {loss:.4f}, AUC: {auc:.4f}, '
          f'F1: {f1:.4f}')