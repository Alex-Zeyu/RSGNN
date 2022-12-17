import sys
import time

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
from copy import deepcopy

import matplotlib.pyplot as plt

class RSGNN:
    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.best_train_loss = 1
        self.best_train_acc = 0,0,0,0
        self.best_test_acc =0,0,0,0
        self.best_adj = None
        self.weights = None
        self.estimator = None
        self.model = model.to(device)
        self.adj = None

    def fit(self, x, adj,
            train_pos_edge_index, train_neg_edge_index,
            test_pos_edge_index, test_neg_edge_index):
        """
        Train src
        Args:
            x: node feature
            adj: the adjacency matrix: torch.tensor
            epoch:
            train_pos_edge: train positive edge
            train_neg_edge: train negative edge
            test_pos_edge: test positive edge
            test_neg_edge: test negative edge

        Returns:

        """
        result = "result_RSGNN.txt"
        self.adj = adj
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
        self.estimator = EstimateAdj(adj, self.args.symmetric).to(self.device)
        self.optimizer_adj = optim.SGD(self.estimator.parameters(),
                                       momentum=0.9,
                                       lr = self.args.lr_adj)

        # Train model
        t_total = time.time()
        for epoch in range(self.args.epochs):
            if self.args.only_sgnn:
                self.train_sgnn(epoch, x,
                                train_pos_edge_index, train_neg_edge_index,
                                test_pos_edge_index, test_neg_edge_index)
            else:
                for i in range(int(self.args.outer_steps)):
                    self.train_adj(epoch, x,
                                   train_pos_edge_index, train_neg_edge_index,
                                   test_pos_edge_index, test_neg_edge_index)

                for i in range(int(self.args.inner_steps)):
                    self.train_sgnn(epoch, x,
                                    train_pos_edge_index, train_neg_edge_index,
                                    test_pos_edge_index, test_neg_edge_index)

        self.model.load_state_dict(self.weights)
        z = self.model(x, self.best_adj)
        auc_test, Binary_f1_test, Micro_f1_test, Macro_f1_test = \
            self.model.test(z, test_pos_edge_index, test_neg_edge_index)
        # with open(result, 'a') as f:
        #     f.write(f'alpha: {args.alpha}, beta: {args.beta}, phi: {args.phi}' )
        #     f.write('\n')
        #     f.write(f"Best auc: {auc_test}, Best F1: {f1_test}")
        #     f.write('\n')
        #     f.write('\n')
        print(f"Best Test auc: {auc_test:.4f}, "
              f"Best Test Binary-F1: {Binary_f1_test:.4f}, "
              f"Best Test Micro-F1: {Micro_f1_test:.4f}, "
              f"Best Test Macro-F1: {Macro_f1_test:.4f}")
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    def train_sgnn(self, epoch, x,
                   train_pos_edge_index, train_neg_edge_index,
                   test_pos_edge_index, test_neg_edge_index):
        if self.args.debug:
            print("\n=== train_model ===")
        self.estimator.normalize()
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        z = self.model(x, self.estimator.estimated_adj)
        loss_train = self.model.loss(z, train_pos_edge_index, train_neg_edge_index)
        loss_train.backward()
        self.optimizer.step()

        self.estimator.estimated_adj.data.clamp_(-1, 1)
        self.model.eval()
        z = self.model(x, self.estimator.estimated_adj)
        loss_train = self.model.loss(z, train_pos_edge_index, train_neg_edge_index)
        auc_train, Binary_f1_train, Micro_f1_train, Macro_f1_train = \
            self.model.test(z, train_pos_edge_index, train_neg_edge_index)
        loss_test = self.model.loss(z, test_pos_edge_index, test_neg_edge_index)
        auc_test, Binary_f1_test, Micro_f1_test, Macro_f1_test = \
            self.model.test(z, test_pos_edge_index, test_neg_edge_index)
        # choose best adj and model weight
        if loss_train < self.best_train_loss:
            self.best_train_loss = loss_train
            self.best_adj = self.estimator.estimated_adj
            self.weights = deepcopy(self.model.state_dict())
            if self.args.debug:
                print(f"\t=== saving current graph adj "
                      f"Train auc:  {auc_train:.4f} "
                      f"Train Binary-F1: {Binary_f1_train:.4f} "
                      f"Train Micro-F1: {Micro_f1_train:.4f} "
                      f"Train Macr-F1: {Macro_f1_train:.4f}")

        if self.args.debug:
            if epoch % 1 == 0:
                print(f"Epoch: {epoch+1:04d}\n"
                      f"loss_train: {loss_train:.4f} "
                      f"Train auc: {auc_train:.4f} "
                      f"Train Binary-F1: {Binary_f1_train:.4f} "
                      f"Train Micro-F1: {Micro_f1_train:.4f} "
                      f"Train Macro-F1: {Macro_f1_train:.4f}\n"
                      f"loss_test: {loss_test:.4f} "
                      f"Test auc: {auc_test:4f} "
                      f"Test Binary-F1: {Binary_f1_test:.4f} "
                      f"Test Micro-F1: {Micro_f1_test:.4f} "
                      f"Test Macro-F1: {Macro_f1_test:.4f}\n"
                      f"time: {time.time() -t:.4f}s")

    def train_adj(self, epoch, x,
                  train_pos_edge_index, train_neg_edge_index,
                  test_pos_edge_index, test_neg_edge_index):
        if self.args.debug:
            print("\n=== train_adj ===")
        self.estimator.normalize()
        t = time.time()
        self.model.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(self.estimator.estimated_adj, 1)
        loss_fro = torch.norm(self.estimator.estimated_adj - self.adj, p ='fro')
        z = self.model(x, self.estimator.estimated_adj)
        loss_sgnn = self.model.loss(z, train_pos_edge_index, train_neg_edge_index)
        loss_neg_balance = - torch.trace(self.estimator.estimated_adj @
                                         self.estimator.estimated_adj @
                                         self.estimator.estimated_adj )
        loss_feature = self.feature_loss(x, self.estimator.estimated_adj)
        loss_symmetric = torch.norm(self.estimator.estimated_adj -
                                    self.estimator.estimated_adj.t(),p= "fro")
        loss_differential = self.args.phi * loss_fro + \
                            self.args.beta * loss_neg_balance + \
                            self.args.alpha * loss_symmetric + \
                            self.args.mu * loss_l1 + \
                            self.args.zeta * loss_sgnn +\
                            loss_feature
        loss_differential.backward()
        self.optimizer_adj.step()

        total_loss =  self.args.phi * loss_fro + \
                      self.args.beta * loss_neg_balance + \
                      self.args.alpha * loss_symmetric + \
                      self.args.mu * loss_l1 + \
                      loss_feature

        self.estimator.estimated_adj.data.copy_(torch.clamp(
            self.estimator.estimated_adj.data, min=-1, max=1))

        self.model.eval()
        self.estimator.normalize()
        z = self.model(x, self.estimator.estimated_adj)
        loss_train = self.model.loss(z, train_pos_edge_index, train_neg_edge_index)
        auc_train, Binary_f1_train, Micro_f1_train, Macro_f1_train = \
            self.model.test(z, train_pos_edge_index, train_neg_edge_index)
        loss_test = self.model.loss(z, test_pos_edge_index, test_neg_edge_index)
        auc_test, Binary_f1_test, Micro_f1_test, Macro_f1_test = \
            self.model.test(z, test_pos_edge_index, test_neg_edge_index)

        # choose best adj and model weight
        if loss_train < self.best_train_loss:
            self.best_train_loss = loss_train
            self.best_train_acc = \
                auc_train, Binary_f1_train, Micro_f1_train, Macro_f1_train
            self.best_adj = self.estimator.estimated_adj
            self.weights = deepcopy(self.model.state_dict())
            if self.args.debug:
                print(f"\t=== saving current graph adj"
                      f"Train auc:  {auc_train:.4f} "
                      f"Train Binary-F1: {Binary_f1_train:.4f} "
                      f"Train Micro-F1: {Micro_f1_train:.4f} "
                      f"Train Macro-F1: {Macro_f1_train:.4f}")

        if self.args.debug:
            print(f"Epoch: {epoch + 1:04d}\n"
                  f"loss_train: {loss_train:.4f} "
                  f"Train auc: {auc_train:.4f} "
                  f"Train Binary-F1: {Binary_f1_train:.4f} "
                  f"Train Micro-F1: {Micro_f1_train:.4f} "
                  f"Train Macro-F1: {Macro_f1_train:.4f}\n"
                  f"loss_test: {loss_test:.4f} "
                  f"Test auc: {auc_test:.4f} "
                  f"Test Binary-F1: {Binary_f1_test:.4f} "
                  f"Test Micro-F1: {Micro_f1_test:.4f} "
                  f"Test Macro-F1: {Macro_f1_test:.4f} "
                  f"\n=============================\n"
                  f"loss_fro: {loss_fro.item():.4f} "
                  f"loss_sgnn: {loss_sgnn.item():.4f} "
                  f"loss_feat: {loss_feature.item():.4f} "
                  f"loss_symmetric: {loss_symmetric.item():.4f} "
                  f"loss_neg_balance: {loss_neg_balance.item():.4f} "
                  f"loss_l1: {loss_l1.item():.4f} "
                  f"loss_total: {total_loss.item():.4f}\n"
                  f"time: {time.time() - t:.4f}s")


    def feature_loss(self, x, estimate_adj):
        """
        return the loss of feature smoothness between
        Args:
            gamma_1:
            gamma_2:

        Returns:

        """
        zeros = estimate_adj.new_zeros(estimate_adj.shape)
        adj_pos = torch.where(estimate_adj >= 0,
                              estimate_adj,zeros)
        adj_neg = torch.where(estimate_adj<= 0,
                              estimate_adj, zeros)
        D_pos = torch.diag(adj_pos.sum(dim=1))
        D_neg = torch.diag(adj_neg.sum(dim=1))
        # L_pos = D_pos - adj_pos
        # L_neg = D_neg - adj_neg
        ret = self.args.gamma_1 * torch.trace(x.T @ (D_pos - adj_pos) @ x) - \
              self.args.gamma_2 * torch.trace(x.T @ (D_neg - adj_neg) @ x)
        return ret

class EstimateAdj(nn.Module):
    def __init__(self, adj, symmetric = True):
        super(EstimateAdj, self).__init__()
        n = len(adj)
        self.estimated_adj = Parameter(torch.FloatTensor(n, n))
        self._init_estimation(adj)
        self.symmetric = symmetric

    def _init_estimation(self, adj):
        with torch.no_grad():
            n = len(adj)
            self.estimated_adj.data.copy_(adj)

    def forward(self):
        return self.estimated_adj

    def normalize(self):
        if self.symmetric:
            self.estimated_adj.data = (self.estimated_adj.data + self.estimated_adj.t()) / 2


    def _normalize(self, mx):
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx



