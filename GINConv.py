import torch
import torch_geometric
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from torch_geometric.utils.random import erdos_renyi_graph
import time
import random
from math import floor, ceil
from copy import deepcopy

from os import path
import bz2
import pickle
import _pickle as cPickle
import sys

class GINConv(torch.nn.Module):
    def __init__(self, node_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(node_size, node_size)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        self.leak_relu1 = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(node_size, node_size)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        self.leak_relu2 = torch.nn.LeakyReLU()

    def forward(self, A, X, E):
        """
        Params
        ------
        A [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix

        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        X = self.linear1(X + A @ X + E)
        X = self.leak_relu1(X)
        X = self.linear2(X)
        X = self.leak_relu2(X)
        return X
