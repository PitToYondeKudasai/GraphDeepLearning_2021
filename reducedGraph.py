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
import importlib
from os import path
import bz2
import pickle
import _pickle as cPickle
import sys

import graph
from graph import Node, Graph
from syntheticGraph import syntheticGraph

importlib.reload(graph)
from graph import Graph, Node

class reducedGraph():
    def __init__(self, originalGraph, coarse_type = 'baseline', n_supernodes = 3):
        self.graph = originalGraph
        coarsening = {'baseline': self.baselineReduction, 'custom':self.customReduction, 'hem':self.heavyEdgeReduction}
        self.n_supernodes = n_supernodes
        self.n_edges = 0
        self.edges = []
        coarsening[coarse_type]()
        self.computeProjection()
        self.compute_reduced()
        # D = torch.diag(torch.sum(self.edge_index, axis = 1))
        # laplacian = D - self.edge_index
        # self.eigenvalues = reducedGraph.eigen_analysis(laplacian)

    # def eigen_analysis(matrix):
    #   ''' Given a matrix, it returns the eigenavlues and aigenvectors sorted in decreasing order '''
    #   values, _ = np.linalg.eig(matrix)
    #   indices = np.argsort(values)
    #   values = torch.tensor(values[indices])
    #   return values


    def heavyEdgeReduction(self):

        (x, y) = np.where(np.triu(self.graph.adjacency_matrix.numpy()))

        while(True):
            tmpGraph = Graph()
            for i in range(self.graph.n_nodes):
                tmpGraph.add_node(Node(i))
            for i in range(x.shape[0]):
                tmpGraph.add_edge(x[i], y[i])

            counter = 0;
            while ((self.graph.n_nodes-counter > self.n_supernodes) and (len(x)-counter >= 0)):
                u_id = np.random.choice(tmpGraph.get_nodes())
                v_id = tmpGraph.get_heavy_edge(u_id)
                tmpGraph.join_nodes(u_id, v_id)
                counter+=1

            mapping = tmpGraph.get_mapping()
            edge_index = self.mappingToAdjacencyMatrix(mapping)
            if np.sum(edge_index) != 0:
                break

        self.mapping = mapping
        self.edge_index = edge_index


    def computeProjection(self):
      '''ex metodo'''
      self.P = torch.zeros((self.n_supernodes, self.graph.n_nodes))
      self.P_plus = torch.zeros((self.graph.n_nodes, self.n_supernodes))
      self.Gamma = torch.zeros((self.n_supernodes, self.n_supernodes))
      for key in self.mapping.keys():
          size_cluster = len(self.mapping[key])
          self.P[key, self.mapping[key]] = 1/size_cluster
          self.P_plus[self.mapping[key], key] = 1
          self.Gamma[key, key] = size_cluster
      self.PI = self.P_plus @ self.P

    def baselineReduction(self):
      ''' Pensato per grafi undirected '''
      counter = 1;
      while True:
          sp_graph = shortest_path(csr_matrix(self.graph.adjacency_matrix), directed=False)
          super_nodes = np.random.choice(np.arange(0,self.graph.n_nodes), self.n_supernodes, replace=False, p=np.ones(self.graph.n_nodes)/self.graph.n_nodes)
          super_nodes.sort()

          sp_graph = sp_graph[:,super_nodes]
          nodes_cluster = np.argmin(sp_graph,axis=1)

          mapping = {}
          for i in range(self.n_supernodes):
              mapping[i] = []
          for i in range(nodes_cluster.shape[0]):
              mapping[nodes_cluster[i]].append(i)


          edge_index = self.mappingToAdjacencyMatrix(mapping)
          if np.sum(edge_index) != 0:
              break

          counter +=1

      self.adjacency_matrix = torch.tensor(edge_index)
      self.mapping = mapping

    def mappingToAdjacencyMatrix(self, mapping):
        self.edges = []
        self.n_edges = 0
        edge_index = np.zeros((self.n_supernodes, self.n_supernodes))
        for i in range(self.n_supernodes):
            for j in range(i, self.n_supernodes):
                found = False
                if(i != j and not found):
                    for l in mapping[i]:
                        for p in mapping[j]:
                            if(self.graph.adjacency_matrix[l,p]!=0):
                                edge_index[i,j] = 1
                                self.edges.append((i,j))
                                edge_index[j,i] = 1
                                self.n_edges += 1
                                found = True
                                break
                        if(found): break
        return edge_index

    def customReduction(self):
      self.adjacency_matrix = torch.tensor([[0,1,1,1],[1,0,0,1],[1,0,0,0],[1,1,0,0]])
      self.mapping = {0:[0, 1, 4], 1:[2, 5, 8], 2:[3], 3:[6, 7]}
      for i in range(self.n_supernodes):
          for j in range(i, self.n_supernodes):
              found = False
              if(i != j and not found):
                  for l in self.mapping[i]:
                      for p in self.mapping[j]:
                          if(self.graph.adjacency_matrix[l,p]!=0):
                              self.edges.append((i,j))
                              found = True
                              break
                      if(found): break


    def compute_reduced(self):
      self.edge_index = torch.zeros((self.n_supernodes, self.n_supernodes))
      for edge in self.edges:
        self.edge_index[edge[0], edge[1]] = torch.sum(self.graph.adjacency_matrix[self.mapping[edge[0]],:][:,self.mapping[edge[1]]])
        self.edge_index[edge[1], edge[0]] = self.edge_index[edge[0], edge[1]]
