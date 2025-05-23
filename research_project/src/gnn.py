import json
import os
import pickle
from collections import Counter

import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.utils.data import random_split
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import GCNConv, global_add_pool


class GraphDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.all_graphs = []

        for file in os.listdir(self.root):
            if file.endswith(".pkl"):
                file_path = os.path.join(self.root, file)
                with open(file_path, "rb") as f:
                    graphs_in_file = pickle.load(f)
                    if isinstance(graphs_in_file, list):
                        for graph_data in graphs_in_file:
                            self.all_graphs.append((graph_data, file_path))
                    else:
                        self.all_graphs.append((graphs_in_file, file_path))

        with open(
            "research_project/tactic_labels/de_dust2/00e7fec9-cee0-430f-80f4-6b50443ceacd.json",
            "r",
        ) as f:
            self.label_mapping = json.load(f)
        self.unique_labels = sorted(set(self.label_mapping.values()))
        self.label_to_id = {label: idx for idx, label in enumerate(self.unique_labels)}

        self.processed_graphs = []
        for graph_data, file_path in self.all_graphs:
            self.processed_graphs.append(
                self._process_graph_data(graph_data, file_path)
            )

    def _process_graph_data(self, raw_data, file_path):
        """Process a single graph's data with its source file information"""
        node_dicts = raw_data["nodes_data"]
        sorted_node_ids = sorted(node_dicts.keys())
        node_id_to_index = {node_id: i for i, node_id in enumerate(sorted_node_ids)}

        node_features = [
            [float(v) for v in node_dicts[nid].values()] for nid in sorted_node_ids
        ]
        x = torch.tensor(node_features, dtype=torch.float)

        edge_list = raw_data["edges_data"]
        edge_index = []
        edge_attr = []

        for src, tgt, attr in edge_list:
            if src in node_id_to_index and tgt in node_id_to_index:
                edge_index.append([node_id_to_index[src], node_id_to_index[tgt]])
                edge_attr.append([attr["dist"]])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        filename = os.path.basename(file_path)
        file_id = filename.replace("graph-rounds-", "").replace(".pkl", "")
        str_label = self.label_mapping.get(file_id, "default_map_control")
        label = self.label_to_id[str_label]
        y = torch.tensor(label, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)

    def len(self):
        return len(self.processed_graphs)

    def get(self, idx):
        return self.processed_graphs[idx]


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)
        return self.lin(x)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data_path = "research_project\graphs\\00e7fec9-cee0-430f-80f4-6b50443ceacd"

    dataset = GraphDataset(data_path)

    print(
        "Label distribution:",
        Counter([dataset[i].y.item() for i in range(len(dataset))]),
    )

    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16)

    print("Train set size:", len(train_set), "Test set size:", len(test_set))

    sample_graph = dataset.get(0)
    input_dim = sample_graph.num_node_features
    output_dim = int(max(graph.y.item() for graph in dataset)) + 1  # num classes

    model = GNN(input_dim=input_dim, hidden_channels=64, output_dim=output_dim).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 31):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            total_correct += (pred == batch.y).sum().item()
            total_samples += batch.y.size(0)

        accuracy = total_correct / total_samples if total_samples > 0 else 0

        print(f"Epoch {epoch}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2%}")

        model.eval()
        predictions = []

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)

                for j in range(data.num_graphs):
                    predictions.append(
                        {
                            "index": i * test_loader.batch_size + j,
                            "true": data.y[j].item(),
                            "pred": pred[j].item(),
                            "x": data.x[data.batch == j].cpu().numpy(),  # node features
                            "edge_index": data.edge_index[:, data.batch == j]
                            .cpu()
                            .numpy(),
                        }
                    )
        return predictions


def interactive_round(pred_data):
    x = pred_data["x"]
    true = pred_data["true"]
    pred = pred_data["pred"]
    df = pd.DataFrame(x, columns=["x", "y", "z", ...])  # trim to 2D if needed

    fig = px.scatter(
        df, x="x", y="y", color_discrete_sequence=["green" if pred == true else "red"]
    )
    fig.update_layout(title=f"Pred: {pred} | True: {true}", width=600, height=500)
    fig.show()


if __name__ == "__main__":
    interactive_round(train())  # Example to visualize the first prediction
