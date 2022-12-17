import sys

import numpy as np
import torch
import os.path as osp
from torch_geometric.datasets import BitcoinOTC,SNAPDataset
from torch_geometric.utils import  degree
import networkx as nx


def bitcoin_alpha(path="./data/soc-sign-bitcoinalpha.csv"):
    edges = np.genfromtxt(fname=path,
                          delimiter=",", dtype=int)
    pos = edges[:,2] >0
    neg = edges[:,2] <0
    return torch.from_numpy(edges[pos, 0:2]).t().long(), \
           torch.from_numpy(edges[neg,0:2]).t().long()

def epinion(path="./data/soc-sign-epinions.txt", num_nodes=20000):
    edges = np.genfromtxt(fname=path,
                          comments="#", delimiter="\t", dtype=int)
    pos = edges[:,2] >0
    neg = edges[:,2] <0
    pos_edges = edges[pos,0:2]
    neg_edges = edges[neg,0:2]
    node_list = list(range(12000))
    pos_graph = nx.Graph()
    neg_graph = nx.Graph()

    pos_graph.add_edges_from(pos_edges.tolist())
    neg_graph.add_edges_from(neg_edges.tolist())

    pos_subgraph = pos_graph.subgraph(node_list)
    neg_subgraph = neg_graph.subgraph(node_list)

    pos_edges = list(pos_subgraph.edges())
    neg_edges = list(neg_subgraph.edges())

    pos_edges = np.array(pos_edges)
    neg_edges = np.array(neg_edges)

    return torch.from_numpy(pos_edges).t().long(), \
           torch.from_numpy(neg_edges).t().long()


def slashdot():
    edges = np.genfromtxt(fname="./data/soc-sign-Slashdot081106.txt",
                          comments="#", delimiter="\t", dtype=int)
    pos = edges[:,2] >0
    neg = edges[:,2] <0

    pos_edges = edges[pos,0:2]
    neg_edges = edges[neg,0:2]
    node_list = list(range(12000))
    pos_graph = nx.Graph()
    neg_graph = nx.Graph()

    pos_graph.add_edges_from(pos_edges.tolist())
    neg_graph.add_edges_from(neg_edges.tolist())

    pos_subgraph = pos_graph.subgraph(node_list)
    neg_subgraph = neg_graph.subgraph(node_list)

    pos_edges = list(pos_subgraph.edges())
    neg_edges = list(neg_subgraph.edges())

    pos_edges = np.array(pos_edges)
    neg_edges = np.array(neg_edges)

    return torch.from_numpy(pos_edges).t().long(), \
           torch.from_numpy(neg_edges).t().long()

def bitcoin_otc():
    name = 'BitcoinOTC-1'
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    dataset = BitcoinOTC(path, edge_window_size=1)
    pos_edge_indices, neg_edge_indices = [], []

    for data in dataset:
        pos_edge_indices.append(data.edge_index[:, data.edge_attr > 0])
        neg_edge_indices.append(data.edge_index[:, data.edge_attr < 0])

    pos_edge_index = torch.cat(pos_edge_indices, dim=1)
    neg_edge_index = torch.cat(neg_edge_indices, dim=1)
    return pos_edge_index, neg_edge_index

