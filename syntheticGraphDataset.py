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

from syntheticGraph import syntheticGraph
from reducedGraph import reducedGraph
from GINConv import GINConv
from GNN import GNN

class syntheticGraphDataset():
  def __init__(self, entireMatrix=False):
    self.entireMatrix = entireMatrix

    # Given and processed data variables
    self.nGraphs = 0
    self.graphs = []
    self.reducedGraphs = []
    self.preprocessedGraphs = {}

    # Supporting variables to compute the loss
    self.w_hatGraphs = []
    self.originalGraphsLoss = []

    # Supporting variables for the dispatcher
    self.batchesIndices = []
    self.lastBatch = 0
    self.lastElementBatch = 0
    self.lastElementBatchSpecific = {}

  def getNextBatch(self, graph=None, size=None):
    givenGraph = graph
    if (graph == None):
      graph = self.batchesIndices[self.lastBatch]
      fromEdge = self.lastElementBatch
    else:
      fromEdge = self.lastElementBatchSpecific[graph]

    if (self.entireMatrix == False):
      if ((size == None) or (size>1)):
        print("Be careful, you are asking me a batch of size bigger than one. Since the dataset is in entireMatrix = false this is impossible. I'll set size=1 for you")
      size = 1
    if (size == None):
      size = self.reducedGraphs[graph].n_edges

    toEdge = min(fromEdge + size, self.reducedGraphs[graph].n_edges)

    if graph not in self.preprocessedGraphs.keys():

      A = []
      X = []
      E = []

      for edge in self.reducedGraphs[graph].edges[fromEdge:toEdge]:
        _A, _X, _E = syntheticGraphDataset.processEdgeReducedGraph(self.graphs[graph], self.reducedGraphs[graph], edge, self.entireMatrix)
        A.append(_A)
        X.append(_X)
        E.append(_E)

    else:
      A = self.preprocessedGraphs[graph]['A'][fromEdge:toEdge]
      X = self.preprocessedGraphs[graph]['X'][fromEdge:toEdge]
      E = self.preprocessedGraphs[graph]['E'][fromEdge:toEdge]

    if (givenGraph == None):
      if (toEdge >= self.reducedGraphs[graph].n_edges):
        self.lastElementBatch = 0
        self.lastBatch +=1
      else:
        self.lastElementBatch += toEdge-fromEdge
    else:
      if (toEdge >= self.reducedGraphs[graph].n_edges):
        self.lastElementBatchSpecific[graph] = 0
      else:
        self.lastElementBatchSpecific[graph] += toEdge-fromEdge

    x = [edge[0] for edge in self.reducedGraphs[graph].edges[fromEdge:toEdge]]
    y = [edge[1] for edge in self.reducedGraphs[graph].edges[fromEdge:toEdge]]
    return torch.stack(A), torch.stack(X), torch.stack(E).unsqueeze(2), graph, x, y

  def graphNumberBatches(self, graph, batchSize):
    return ceil(self.reducedGraphs[graph].n_edges/batchSize)

  def resetDispatcher(self):
    self.batchesIndices = [i for i in range(len(self.batchesIndices))]
    self.lastElementBatch = 0
    self.lastBatch = 0

  def shuffle(self):
    random.shuffle(self.batchesIndices)

  def addGraph(self, graph, reducedGraph, n_eig, preprocessData = False):
    graphIndex = self.nGraphs
    self.graphs.append(graph)
    self.reducedGraphs.append(reducedGraph)

    if (preprocessData):
      A, X, E = syntheticGraphDataset.processEntireReducedGraph(graph, reducedGraph, self.entireMatrix)
      self.preprocessedGraphs[graphIndex]= {}
      self.preprocessedGraphs[graphIndex]['A'] = A
      self.preprocessedGraphs[graphIndex]['X'] = X
      self.preprocessedGraphs[graphIndex]['E'] = E

    self.batchesIndices.append(graphIndex)
    self.lastElementBatchSpecific[graphIndex] = 0

    self.w_hatGraphs.append(torch.zeros((reducedGraph.n_supernodes, reducedGraph.n_supernodes)))
    self.w_hatGraphs[graphIndex] = self.reducedGraphs[graphIndex].edge_index
    self.originalGraphsLoss.append(self.rayleigh_loss(graphIndex, n_eig))
    self.w_hatGraphs[graphIndex] = torch.zeros((reducedGraph.n_supernodes, reducedGraph.n_supernodes))

    self.nGraphs += 1

  def generateGraph(size, name_graph_class):
    graph = syntheticGraph(name_graph_class, size)
    return graph

  def reduceGraph(graph, coarse_type, n_supernodes, ):
    reducedGraph = reducedGraph(graph, coarse_type, n_supernodes)
    return reducedGraph

  def computeBatch(graph, node_list, first_mapping, second_mapping, entireMatrix):
    if (entireMatrix):
      A = graph.adjacency_matrix
      A[node_list,:] = 0
      A[:, node_list] = 0

      X = graph.node_features
      X[node_list,:] = 0

      E = graph.edge_weights
      E[node_list, :] = 0
      E = torch.sum(E, axis=1)

    else:
      A = graph.adjacency_matrix[node_list,:][:,node_list]
      X = graph.node_features[node_list,:]
      E = torch.sum(graph.edge_weights[node_list,:], axis=1)

    return A,X,E

  def processEntireReducedGraph(graph, reducedGraph, entireMatrix=False):
    if (entireMatrix):
      entire_node_set = set(np.arange(0, graph.n_nodes))

    A = []
    X = []
    E = []
    processedData = []
    for edge in reducedGraph.edges:
      if (not entireMatrix):
        node_list = reducedGraph.mapping[edge[0]] + reducedGraph.mapping[edge[1]]
        node_list.sort()
      else:
        node_list = list(entire_node_set - set(reducedGraph.mapping[edge[0]] + reducedGraph.mapping[edge[1]]))

      _A, _X, _E = syntheticGraphDataset.computeBatch(graph, node_list, reducedGraph.mapping[edge[0]], reducedGraph.mapping[edge[1]], entireMatrix)
      A.append(_A)
      X.append(_X)
      E.append(_E)

    return A, X, E

  def processEdgeReducedGraph(graph, reducedGraph, edge, entireMatrix=False):
    ''' Edge is a tuple (i,j) referring to the super edge from which you wanto to precess the data. reducedGraph.edges should contain it too.'''
    if (entireMatrix):
      entire_node_set = set(np.arange(0, graph.n_nodes))
      node_list = list(entire_node_set - set(reducedGraph.mapping[edge[0]] + reducedGraph.mapping[edge[1]]))
    else:
      node_list = reducedGraph.mapping[edge[0]] + reducedGraph.mapping[edge[1]]
      node_list.sort()

    A, X, E = syntheticGraphDataset.computeBatch(graph, node_list, reducedGraph.mapping[edge[0]], reducedGraph.mapping[edge[1]], entireMatrix)
    return A, X, E

  def store_w_hat(self, graph, values, x, y):
    '''
      values is a tensor of size nx1 (output of the model)
      x is a list nx1
      y is a list nx1
    '''
    w_hat = self.w_hatGraphs[graph]
    values = values.squeeze()
    w_hat[x,y] = values
    w_hat[y,x] = values

  def reset_w_hat(self, graph= None):
    reducedGraphs = self.reducedGraphs
    w_hat = self.w_hatGraphs

    if (not (graph == None)):
      w_hat[graph] = torch.zeros((reducedGraphs[graph].n_supernodes, reducedGraphs[graph].n_supernodes))
    else:
      for graph in range(self.nGraphs):
        w_hat[graph] = torch.zeros((reducedGraphs[graph].n_supernodes, reducedGraphs[graph].n_supernodes))

  def rayleigh_loss(self, graphIndex, n_eig):

    w_hat = self.w_hatGraphs[graphIndex]
    reducedGraph = self.reducedGraphs[graphIndex]
    graph = self.graphs[graphIndex]
    d_hat = torch.diag(torch.sum(w_hat, axis = 1))
    gamma_prime = torch.diag(torch.pow(torch.diag(reducedGraph.Gamma), -0.5))
    gamma_prime = gamma_prime
    L_hat = gamma_prime @ (d_hat - w_hat) @ gamma_prime
    P_italic = gamma_prime @ reducedGraph.P_plus.T
    P_eig = P_italic @ graph.eigenvectors
    loss = 0

    for i in range(n_eig):
      rayleigh_original = ((graph.eigenvectors[:,i].T @ graph.laplacian @ graph.eigenvectors[:,i])/
                            graph.eigenvectors[:,i].T @ graph.eigenvectors[:,i])
      rayleigh_reconstruct = ((P_eig[:,i].T @ L_hat @ P_eig[:,i])/(P_eig[:,i].T @ P_eig[:,i]))
      loss += torch.abs(rayleigh_original - rayleigh_reconstruct)/n_eig

    return loss

# Pickle a file and then compress it into a file with extension

  def export_dataset(title, data):
     with bz2.BZ2File(title , 'w') as f:
      cPickle.dump(data, f)

     # Load any compressed pickle file
  def decompress_pickle(file):
     data = bz2.BZ2File(file, 'rb')
     data = cPickle.load(data)
     return data

  def import_dataset(file):
    #TRAINING SET
    start = time.time()
    print("Loading the compressed set...")
    training_set = syntheticGraphDataset.decompress_pickle(file)
    print("Dataset loaded in ", time.time()-start, "seconds\n")
    return training_set
