# import library
import os.path as osp
import sys
import time
import argparse
import numpy as np
import torch

from SGCN import SignedGCN
from rsgnn import RSGNN
from utils import create_spectral_features, split_edges,\
    random_sign_pertubation,random_pos_pertubation,random_neg_pertubation
from torch_geometric import seed_everything
from torch_geometric.datasets import BitcoinOTC,SNAPDataset
from torch_geometric.utils import coalesce,to_undirected,to_dense_adj
from data_preparation import bitcoin_alpha, bitcoin_otc, slashdot, epinion


# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true',default=False,
                    help='debug mode')
parser.add_argument('--only_sgnn', action='store_true',default=False,
                    help='test the performance of SGNN without other components')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=2022,
                    help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default= 64,
                    help='Number of hidden units.')
parser.add_argument('--layers', type=int, default=2,
                    help='Number of model layers.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='BitcoinOTC',
        choices=['BitcoinOTC', 'BitcoinAlpha', 'Epinions', 'Slashdot'],
                    help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.1,
                    help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=400,
                    help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=1,
                    help='weight of symmetric loss')
parser.add_argument('--beta', type=float, default=1,
                    help='weight of negative balance degree')
parser.add_argument('--gamma_1', type=float, default=1,
                    help='weight of feature smoothness loss')
parser.add_argument('--gamma_2', type=float, default=2,
                    help='weight of feature distinction loss')
parser.add_argument('--phi', type=float, default=5e-4,
                    help='weight of l2 norm')
parser.add_argument('--mu', type=float, default=5e-4,
                    help='weight of l1 norm')
parser.add_argument('--zeta', type=float, default=0,
                    help='weight of sgnn loss')
parser.add_argument('--inner_steps', type=int, default=2,
                    help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1,
                    help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.005,
                    help='lr for training adj')
parser.add_argument('--symmetric', action='store_true', default=False,
                    help='whether use symmetric matrix')
parser.add_argument('--noise', type=str, default='sp',
        choices=['sp', 'pp', 'np'], help='dataset')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")



if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

seed_everything(args.seed)

# name = 'BitcoinOTC-1'
# path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
# dataset = BitcoinOTC(path, edge_window_size=1)
#
# pos_edge_indices, neg_edge_indices = [], []
#
# for data in dataset:
#     pos_edge_indices.append(data.edge_index[:, data.edge_attr > 0])
#     neg_edge_indices.append(data.edge_index[:, data.edge_attr < 0])
#
# pos_edge_index = torch.cat(pos_edge_indices, dim=1).to(device)
# neg_edge_index = torch.cat(neg_edge_indices, dim=1).to(device)

if args.dataset == "BitcoinOTC":
    pos_edge_index, neg_edge_index = bitcoin_otc()
elif args.dataset == "BitcoinAlpha":
    pos_edge_index, neg_edge_index = bitcoin_alpha()
elif args.dataset == "Epinions":
    pos_edge_index, neg_edge_index = epinion()
elif args.dataset == "Slashdot":
    pos_edge_index, neg_edge_index = slashdot()

pos_edge_index = pos_edge_index.to(device)
neg_edge_index = neg_edge_index.to(device)

train_pos_edge_index, test_pos_edge_index = split_edges(pos_edge_index)
train_neg_edge_index, test_neg_edge_index = split_edges(neg_edge_index)

if args.noise == "sp":
    train_pos_edge_index, train_neg_edge_index = \
        random_sign_pertubation((train_pos_edge_index, train_neg_edge_index),
                            ratio= args.ptb_rate)
elif args.noise == "pp":
    train_pos_edge_index, train_neg_edge_index = \
        random_pos_pertubation((train_pos_edge_index, train_neg_edge_index),
                            ratio= args.ptb_rate)
elif args.noise == "np":
    train_pos_edge_index, train_neg_edge_index = \
        random_neg_pertubation((train_pos_edge_index, train_neg_edge_index),
                            ratio= args.ptb_rate)

train_pos_edge_attr = torch.ones(train_pos_edge_index.shape[1])
train_neg_edge_attr = torch.ones(train_neg_edge_index.shape[1])* -1

train_edge = torch.cat([train_pos_edge_index, train_neg_edge_index], dim =1)
train_edge_attr = torch.cat([train_pos_edge_attr, train_neg_edge_attr]).to(device)

train_edge, train_edge_attr = coalesce(train_edge, train_edge_attr)

train_edge, train_edge_attr = to_undirected(train_edge, train_edge_attr)
train_edge_attr = train_edge_attr.clamp(min=-1, max=1)


adj = to_dense_adj(edge_index=train_edge, edge_attr=train_edge_attr).squeeze().to(device)

x = create_spectral_features(train_pos_edge_index, train_neg_edge_index).to(device)

model = SignedGCN(in_channels= x.shape[1], hidden_channels=args.hidden,
                  num_layers=args.layers)


rsgnn = RSGNN(model, args, device)

if args.only_sgnn:
    rsgnn.fit(x, adj,
              train_pos_edge_index, train_neg_edge_index,
              test_pos_edge_index, test_neg_edge_index)
else:
    rsgnn.fit(x, adj,
              train_pos_edge_index, train_neg_edge_index,
              test_pos_edge_index, test_neg_edge_index)




