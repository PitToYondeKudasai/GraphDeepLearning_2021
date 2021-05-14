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
from torch_geometric.utils.random import erdos_renyi_graph, barabasi_albert_graph
import time
import random
from math import floor, ceil
from copy import deepcopy

from os import path
import bz2
import pickle
import _pickle as cPickle
import sys


class syntheticGraph():
    def __init__(self, size = 10, p=0.1, name_graph_class = 'erdos_renyi_graph', custom_graph=None):
      self.size = size
      self.p = p
      self.name_graph_class = name_graph_class
      self.adjacency_matrix = custom_graph
      create_graph = {'erdos_renyi_graph' : self.erdos_renyi_graph, 
                      'custom_graph':self.custom_graph, 
                      'barabasi_albert_graph':self.barabasi_albert_graph}

      create_graph[name_graph_class]()
      self.n_nodes = self.adjacency_matrix.shape[0]
      self.compute_nodes_features()
      self.edge_weights = self.adjacency_matrix
      self.laplacian = torch.diag(torch.sum(self.edge_weights, axis = 1)) - self.edge_weights
      self.eigenvalues, self.eigenvectors = syntheticGraph.eigen_analysis(self.laplacian)

    def barabasi_albert_graph(self):
      if (self.size == None ):
        raise Exception('Ehi I need the size to create arabasi-albert graph.')
      self.adjacency_matrix = torch_geometric.utils.to_dense_adj(barabasi_albert_graph(self.size, 4))[0]

    def erdos_renyi_graph(self):
      if (self.p == None or self.size == None ):
        raise Exception('Ehi I need the size and the probability for the edge size,p  to create erdos renyi graph.')
      self.adjacency_matrix = torch_geometric.utils.to_dense_adj(erdos_renyi_graph(self.size, self.p))[0]

    def custom_graph(self):
      if (self.adjacency_matrix == None):
        raise Exception('Ehi I need a custom adjacency matrix to create a custom graph.')
      
      adj_list = np.where(self.adjacency_matrix)
      self.adjacency_matrix = torch_geometric.utils.to_dense_adj(torch.tensor(adj_list))[0]


    def get_adj_list(adj):
      ''' Given an adjacency matrix it returns a dictionary with the adjacency list '''
      n = adj.shape[0]
      adj_list = dict((k, []) for k in range(n))
      pos_x, pos_y = np.where(adj != 0)
      for i in range(len(pos_x)):
          adj_list[pos_x[i]].append(pos_y[i])
      return adj_list

    def compute_nodes_features(self):
      self.node_features = []
      degree = np.sum(self.adjacency_matrix.numpy(), axis = 1)
      adj_dict = syntheticGraph.get_adj_list(self.adjacency_matrix)
      node_features = []
      for node in range(self.n_nodes):
        neighborhood_degree = degree[adj_dict[node]]
        self.node_features.append([degree[node],
                                   min(neighborhood_degree, default = 0),
                                   max(neighborhood_degree, default = 0),
                                   np.mean(neighborhood_degree) if len(neighborhood_degree) != 0 else 0,
                                   np.std(neighborhood_degree) if len(neighborhood_degree) != 0 else 0])
      self.node_features=torch.tensor(self.node_features)


    def eigen_analysis(matrix):
      ''' Given a matrix, it returns the eigenavlues and aigenvectors sorted in decreasing order '''
      values, vectors = np.linalg.eig(matrix)
      indices = np.argsort(values)
      values = torch.tensor(values[indices])
      vectors = torch.tensor(vectors[:, indices])
      return values, vectors
