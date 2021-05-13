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

from GINConv import GINConv


class GNN(torch.nn.Module):
    def __init__(self, node_size, n_layers):
         super().__init__()
         self.linear_edges = torch.nn.Linear(1, node_size)
         self.linear_nodes = torch.nn.Linear(5, node_size)

         self.convs = torch.nn.ModuleList()
         for _ in range(n_layers):
                 self.convs.append(GINConv(node_size))
         self.linear = torch.nn.Linear(node_size, 1)
         self.leak_relu = torch.nn.LeakyReLU()

    def forward(self, A, X, E):

         E = self.linear_edges(E)
         X = self.linear_nodes(X)

         for layer in self.convs:
             X = layer(A, X, E)
         X = self.linear(X.mean(axis = 1))
         X = 1 + self.leak_relu(X) #torch.nn.functional.relu(X)
         return X

    def evaluateRayleighLoss(self, dataset, hyperparams, verbose=False):
        if verbose: print("Rayleigh evaluation started..")
        with torch.no_grad():
          improvements = []
          dataset.reset_w_hat()
          dataset.resetDispatcher()
          for graph in dataset.batchesIndices:
            if verbose: print("Graph ", graph, end=" ")
            start = time.time()
            loss = 0
            for batch in range(dataset.graphNumberBatches(graph, hyperparams['batch_size'])):
              A, X, E, _, x, y = dataset.getNextBatch(graph, hyperparams['batch_size'])
              out = self(A, X, E)
              dataset.store_w_hat(graph, out, x, y)
            loss = dataset.rayleigh_loss(graph, hyperparams['n_eig'])
            imp = (dataset.originalGraphsRayLeighLoss[graph] - loss)/dataset.originalGraphsRayLeighLoss[graph]
            improvements.append(imp)
            if verbose: print("--- relative improvement percentage ", imp.item()*100, "%")
        return improvements

    def evaluateEigenError(self, dataset, hyperparams, verbose=False):
      if verbose: print("Eigenerror valuation started..")
      with torch.no_grad():
        improvements = []
        dataset.reset_w_hat()
        dataset.resetDispatcher()
        for graph in dataset.batchesIndices:
          if verbose: print("Graph ", graph, end=" ")
          start = time.time()
          loss = 0
          for batch in range(dataset.graphNumberBatches(graph, hyperparams['batch_size'])):
            A, X, E, _, x, y = dataset.getNextBatch(graph, hyperparams['batch_size'])
            out = self(A, X, E)
            dataset.store_w_hat(graph, out, x, y)
          eigenerror = dataset.eigenError(graph, hyperparams['n_eig'])
          imp = (dataset.originalEigenError[graph] - eigenerror)/dataset.originalEigenError[graph]
          improvements.append(imp)
          if verbose: print("--- relative improvement percentage ", imp.item()*100, "%")
      return improvements
