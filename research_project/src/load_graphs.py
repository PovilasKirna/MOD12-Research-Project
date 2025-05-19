import pickle
import torch
from torch_geometric.data import Data

def load_graph_file(path):
    """Load all graph dictionaries from a .pkl file."""
    with open(path, 'rb') as f:
        graphs = pickle.load(f)
    return graphs

def convert_to_pyg(graph_dict):
    """Convert a single graph dictionary to a PyG Data object."""
    nodes = graph_dict["nodes_data"]
    edges = graph_dict["edges_data"]

    # Sort node keys for consistent order
    node_keys = sorted(nodes.keys())
    node_features = [list(nodes[k].values()) for k in node_keys]
    x = torch.tensor(node_features, dtype=torch.float)

    edge_index = []
    edge_attr = []

    for src, tgt, attr in edges:
        edge_index.append([src, tgt])
        edge_attr.append([attr["dist"]])  # Just the distance; expand as needed

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Optional label
    if "label" in graph_dict["graph_data"]:
        data.y = torch.tensor([graph_dict["graph_data"]["label"]], dtype=torch.long)

    return data

def graph_generator(pkl_path):
    """Yield one PyG Data object at a time from the .pkl file."""
    graphs = load_graph_file(pkl_path)
    for g in graphs:
        yield convert_to_pyg(g)
