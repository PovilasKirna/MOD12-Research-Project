import os
import pickle
from pathlib import Path

import networkx as nx
import dgl
import numpy as np
import torch as th
from dgl.data import DGLDataset, QM7bDataset
from dgl.data.utils import load_graphs, save_graphs

import create_graphs
import stats


class EstaDataset(DGLDataset):
  """ Template for customizing graph datasets in DGL.

  Parameters
  ----------
  url : str
      URL to download the raw dataset
  raw_dir : str
      Specifying the directory that will store the
      downloaded data or the directory that
      already stores the input data.
      Default: ~/.dgl/
  save_dir : str
      Directory to save the processed dataset.
      Default: the value of `raw_dir`
  force_reload : bool
      Whether to reload the dataset. Default: False
  verbose : bool
      Whether to print out progress information
  """
  def __init__(self,
               url=None,
               raw_dir=None,
               save_dir=None,
               force_reload=False,
               verbose=False):
    super(EstaDataset, self).__init__(name='esta',
                                    url=url,
                                    raw_dir=raw_dir,
                                    save_dir=save_dir,
                                    force_reload=force_reload,
                                    verbose=verbose)

  def download(self):
    # download raw data to local disk
    pass

  def process(self):
    graph_dicts = []
    for file_round in os.listdir(self.raw_dir):
      filename = os.path.join(self.raw_dir, file_round)
      if not filename.endswith(".pkl"): # skip other failes than pkl files
        continue
      with open(filename, 'rb') as f:
        temp_graphs = pickle.load(f)
        graph_dicts.extend(temp_graphs)

    # convenience debug function for building weapon id mapping in create_graphs.py
    # self._print_unique_weapons(graph_dicts)

    # Create graphs from dictionary data
    # TODO: probably better to straight create DGL graphs, but the interface for nx is nicer
    #       but must ensure correct numbering and ids, to keep edges and nodes consistent with attributes
    nxgraphs = []
    for g in graph_dicts:
      graph = nx.DiGraph(**g["graph_data"])
      graph.add_nodes_from([(key,self._num_to_float(node_data)) for (key,node_data) in g["nodes_data"].items()])
      graph.add_edges_from(g["edges_data"])
      nxgraphs.append(graph)

    node_attributes = list(create_graphs.KEYS_PER_NODE)
    edge_attributes = ["dist",]
    # process data to a list of graphs and a list of labels
    self.graphs = [dgl.from_networkx(nx.convert_node_labels_to_integers(graph), node_attrs=node_attributes, edge_attrs=edge_attributes) for graph in nxgraphs]
    self.label = th.zeros([len(nxgraphs), 1], dtype=th.float32)

    # generate attribute feature matrix
    for g in self.graphs:
      # node attributes
      node_attribute_matrix = th.zeros(g.num_nodes(), len(create_graphs.KEYS_PER_NODE), dtype=th.float32)
      for i in range(g.num_nodes()):
        node_attribute_matrix[i] = th.tensor([g.ndata[key][i] for key in g.ndata.keys()], dtype=th.float32)
      g.ndata["attr"] = node_attribute_matrix

      # edge attributes
      edge_attribute_matrix = th.zeros(g.num_edges(), 1, dtype=th.float32)
      for i in range(g.num_edges()):
        edge_attribute_matrix[i] = th.tensor(g.edata["dist"][i], dtype=th.float32)
      g.edata["attr"] = edge_attribute_matrix



  def _num_to_float(self, node_data: dict):
    # convert all relevant fields to floats
    KEYS_FLOATS = ("x", "y", "z", "velocityX", "velocityY", "velocityZ", "viewX", "viewY")
    for key in KEYS_FLOATS:
      node_data[key] = float(node_data[key])
    return node_data

  def _print_unique_weapons(self, graph_dicts):
    weapons = []
    for g in graph_dicts:
      for n in g["nodes_data"].values():
        weapons.append(n["activeWeapon"])
    weapons = set(weapons)
    print(weapons)

  def __getitem__(self, idx):
    """ Get graph and label by index

    Parameters
    ----------
    idx : int
        Item index

    Returns
    -------
    (dgl.DGLGraph, Tensor)
    """
    return self.graphs[idx], self.label[idx]

  def __len__(self):
    """Number of graphs in the dataset"""
    return len(self.graphs)


  def save(self):
    """save the graph list and the labels"""
    graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
    save_graphs(str(graph_path), self.graphs, {'labels': self.label})


  def has_cache(self):
    graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
    return os.path.exists(graph_path)


  def load(self):
    graphs, label_dict = load_graphs(os.path.join(self.save_path, 'dgl_graph.bin'))
    self.graphs = graphs
    self.label = label_dict['labels']


  @property
  def num_labels(self):
    """Number of labels for each graph, i.e. number of prediction tasks."""
    return 0



if __name__ == "__main__":
  # test loading dataset
  #test = QM7bDataset()
  data_folder = Path(__file__).parent / "../../../graphs/" / stats.EXAMPLE_DEMO_PATH.stem
  data_folder = str(data_folder.resolve())
  dataset = EstaDataset(raw_dir=data_folder, force_reload=True)
  print("Number of graphs:", len(dataset))
  g = dataset[0]
  graphs, labels = map(list, zip(*dataset))
  print(g)
