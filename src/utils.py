# import library
import torch
from torch_geometric.utils import degree, \
    to_undirected,negative_sampling,dropout_adj
import numpy as np
from scipy.sparse import csr_matrix, spdiags, find
from numpy.linalg import norm
from tqdm import tqdm
import networkx as nx
from networkx.algorithms.cycles import cycle_basis
from scipy.sparse import csr_matrix
from torch_sparse import coalesce
import scipy.sparse

def create_spectral_features(pos_edge_index, neg_edge_index,
                             in_channels =64,num_nodes=None):
    from sklearn.decomposition import TruncatedSVD

    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    N = edge_index.max().item() + 1 if num_nodes is None else num_nodes
    edge_index = edge_index.to(torch.device('cpu'))

    pos_val = torch.full((pos_edge_index.size(1),), 2, dtype=torch.float)
    neg_val = torch.full((neg_edge_index.size(1),), 0, dtype=torch.float)
    val = torch.cat([pos_val, neg_val], dim=0)

    row, col = edge_index
    edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
    val = torch.cat([val, val], dim=0)

    edge_index, val = coalesce(edge_index, val, N, N)
    val = val - 1

    # Borrowed from:
    # https://github.com/benedekrozemberczki/SGCN/blob/master/src/utils.py
    edge_index = edge_index.detach().numpy()
    val = val.detach().numpy()
    A = scipy.sparse.coo_matrix((val, edge_index), shape=(N, N))
    svd = TruncatedSVD(n_components=in_channels, n_iter=128)
    svd.fit(A)
    x = svd.components_.T
    return torch.from_numpy(x).to(torch.float).to(pos_edge_index.device)


def split_edges(edge_index, test_ratio = 0.2):
    mask = torch.ones(edge_index.size(1), dtype=torch.bool)
    mask[torch.randperm(mask.size(0))[:int(test_ratio * mask.size(0))]] = 0

    train_edge_index = edge_index[:, mask]
    test_edge_index = edge_index[:, ~mask]

    while train_edge_index.max() != edge_index.max():
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        mask[torch.randperm(mask.size(0))[:int(test_ratio * mask.size(0))]] = 0

        train_edge_index = edge_index[:, mask]
        test_edge_index = edge_index[:, ~mask]
    return train_edge_index, test_edge_index



def read_from_file(filepath, delimiter=' ', directed = False):
    '''
    read from the filepath to
    :param filepath: edge path
    :param delimiter:
    :param directed:
    :return: return Sparse adjacency matrix, positive edge indices,
    negative edge indices
    '''
    edges = np.loadtxt(filepath, dtype=int, comments='#')
    positive_index = edges[:,2]>0
    negative_index = edges[:,2]<0
    # m, n = edges.shape
    #
    # rows = edges[:, 0]
    # cols = edges[:, 1]
    # data = edges[:, 2]


    # n_node = int(np.amax(edges[:, 0:2]) + 1) # number of nodes
    # A = csr_matrix((data, (rows, cols)), shape=(n_node, n_node))

    return edges[positive_index][:,0:2], edges[negative_index][:,0:2]



def calculate_triangle_index(edges):
    '''

    :param edges: edges list , edge --> [src, dst, sign]
    :return: triange index
    '''
    sign_dict = dict()
    G = nx.Graph()
    for src, dst, sign in edges:
        sign_dict[(src,dst)] = sign
        sign_dict[(dst,src)] = sign
        G.add_edge(src,dst)
    circles = cycle_basis(G)
    triangle_circles = [circle for circle in circles if len(circle)==3]
    triangle_circles_count = len(triangle_circles)
    balance_triangle = []
    for tri in triangle_circles:
        if sign_dict[tri[0],tri[1]]*sign_dict[tri[1],tri[2]]*sign_dict[tri[2],tri[0]]>0:
            balance_triangle.append(tri)
    balance_triangle_count = len(balance_triangle)

    return balance_triangle_count/triangle_circles_count

def triangle_index(pos_edges, neg_edges):
    '''
    Calculate triangle index of signed graph
    Args:
        pos_edges: positive index
        neg_edges: negative index

    Returns: triangle index

    '''
    pos_edges = to_undirected(pos_edges)
    neg_edges = to_undirected(neg_edges)

    pos_attr = pos_edges.new_ones(pos_edges.shape[1],dtype = pos_edges.dtype)
    neg_attr = neg_edges.new_ones(neg_edges.shape[1],dtype = neg_edges.dtype)*-1

    edge_index = torch.cat([pos_edges, neg_edges],dim=1)
    edge_attr  = torch.cat([pos_attr, neg_attr])

    node_count = edge_index.max()+1
    edge_index, edge_attr = coalesce(index=edge_index, value=edge_attr,
                                     m = node_count, n = node_count, op='mean')
    adj = torch.sparse_coo_tensor(edge_index, edge_attr,
                                  [node_count, node_count]).to_dense().cuda()
    # adj_abs = torch.abs(adj)
    # adj_3 = adj @ adj @ adj
    # adj_3_abs = adj_abs @ adj_abs @ adj_abs
    return adj @ adj @ adj

import torch
from torch_geometric.utils import degree, \
    to_undirected,negative_sampling,dropout_adj
import numpy as np
from scipy.sparse import csr_matrix, spdiags, find
from numpy.linalg import norm
from tqdm import tqdm
import networkx as nx
from networkx.algorithms.cycles import cycle_basis
from scipy.sparse import csr_matrix
from torch_sparse import coalesce
import scipy.sparse


def calculate_walk_balance(edges):
    row = edges[:,0].reshape(-1)
    col = edges[:,1].reshape(-1)
    data = edges[:,2].reshape(-1)
    sparse_adj = csr_matrix((data,(row,col))).toarray()
    return (np.trace(np.exp(sparse_adj))/np.trace(np.exp(np.abs(sparse_adj)))+1)/2

def random_sign_pertubation(edges, ratio):
    '''
    Change the sign of the chosen edge
    :param edges: (pos_edge_index, neg_edge_index)
    :param ratio: edge pertubated edges
    :return: edges (modified_pos_edge_index, modified_neg_edge_index)
    '''
    pos_edge_index, neg_edge_index = edges

    pos_index_mask = torch.empty(pos_edge_index.shape[1]).bernoulli_(p=ratio).long().bool()
    neg_index_mask = torch.empty(neg_edge_index.shape[1]).bernoulli_(p=ratio).long().bool()

    modified_pos_edge_index = torch.cat([neg_edge_index[:,neg_index_mask],
                                         pos_edge_index[:,(pos_index_mask * -1 + 1).bool()]], dim=1)
    modified_neg_edge_index = torch.cat([pos_edge_index[:,pos_index_mask],
                                         neg_edge_index[:,(neg_index_mask * -1 + 1).bool()]], dim=1)

    return modified_pos_edge_index, modified_neg_edge_index

def random_pos_pertubation(edges, ratio):
    pos_edge_index, neg_edge_index = edges
    pos_index_mask = torch.empty(pos_edge_index.shape[1]).bernoulli_(p=ratio).long().bool()
    modified_pos_edge_index = pos_edge_index[:, (pos_index_mask * -1 + 1).bool()]
    modified_neg_edge_index = torch.cat([pos_edge_index[:, pos_index_mask], neg_edge_index], dim=1)
    return modified_pos_edge_index, modified_neg_edge_index

def random_neg_pertubation(edges, ratio):
    pos_edge_index, neg_edge_index = edges
    edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    new_neg_edges = negative_sampling(edges, num_neg_samples=round(edges.shape[1]*ratio))
    modified_pos_edge_index = pos_edge_index
    modified_neg_edge_index = torch.cat([neg_edge_index, new_neg_edges], dim=1)
    return modified_pos_edge_index, modified_neg_edge_index





