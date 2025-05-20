import os
import pickle
from collections import Counter

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.utils.data import random_split
from torch_geometric.data import Data, DataLoader, Dataset
from torch_geometric.nn import GCNConv, global_add_pool


class GraphDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.graph_files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".pkl")
        ]

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        with open(self.graph_files[idx], "rb") as f:
            raw_list = pickle.load(f)

        raw_data = raw_list[0] if isinstance(raw_list, list) else raw_list

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

        label = raw_data["graph_data"].get("strategy_used", 0)
        y = torch.tensor(label, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)


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
        Counter([dataset.get(i).y.item() for i in range(len(dataset))]),
    )

    train_len = int(0.8 * len(dataset))
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])

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

        # Optional: evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)

        test_acc = correct / total if total > 0 else 0

        print(f" → Test Accuracy: {test_acc:.2%}")


if __name__ == "__main__":
    train()
