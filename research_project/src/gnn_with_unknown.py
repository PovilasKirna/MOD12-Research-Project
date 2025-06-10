import os
import pickle
from collections import Counter

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import Dropout, Linear
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import add_self_loops
from torchmetrics import Accuracy, F1Score, Precision, Recall


class GraphDataset(Dataset):
    def __init__(self, graph_root_dir, area_encoder=None, label_to_id=None):
        super().__init__()
        self.graph_root_dir = graph_root_dir
        self.all_graphs = []
        self.area_ids = []

        # Recursively search all folders for .pkl files
        for root, _, files in os.walk(self.graph_root_dir):
            for file in files:
                if file.endswith(".pkl"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "rb") as f:
                        graphs_in_file = pickle.load(f)
                        if isinstance(graphs_in_file, list):
                            for i, graph_data in enumerate(graphs_in_file):
                                self.all_graphs.append((graph_data, file_path, i))
                        else:
                            self.all_graphs.append((graphs_in_file, file_path, 0))

        # Collect areaId for OneHotEncoder
        for graph_data, _, _ in self.all_graphs:
            for node_data in graph_data["nodes_data"].values():
                self.area_ids.append([node_data.get("areaId", 0)])

        if area_encoder is not None:
            self.area_encoder = area_encoder
        else:
            self.area_encoder = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            )
            self.area_encoder.fit(self.area_ids)

        # Normalize position values
        self.all_x = []
        self.all_y = []
        for graph_data, _, _ in self.all_graphs:
            for node in graph_data["nodes_data"].values():
                self.all_x.append(node.get("x", 0))
                self.all_y.append(node.get("y", 0))

        self.global_min_x, self.global_max_x = min(self.all_x), max(self.all_x)
        self.global_min_y, self.global_max_y = min(self.all_y), max(self.all_y)
        self.global_x_range = (
            self.global_max_x - self.global_min_x
            if self.global_max_x != self.global_min_x
            else 1
        )
        self.global_y_range = (
            self.global_max_y - self.global_min_y
            if self.global_max_y != self.global_min_y
            else 1
        )

        # Collect unique labels from all graphs
        if label_to_id is not None:
            self.label_to_id = label_to_id
        else:
            strategies = {
                graph_data.get("graph_data", {}).get("strategy_used", "unknown")
                for graph_data, _, _ in self.all_graphs
            }
            self.label_to_id = {
                label: idx for idx, label in enumerate(sorted(strategies))
            }

        # Convert each graph to a PyG Data object
        self.processed_graphs = [
            self._process_graph_data(graph_data, file_path, idx)
            for graph_data, file_path, idx in self.all_graphs
        ]

    def __len__(self):
        return len(self.processed_graphs)

    def __getitem__(self, idx):
        return self.processed_graphs[idx]

    def _process_graph_data(self, graph_dict, file_path, graph_idx):
        # selected_keys = ["x", "y", "hp", "armor", "isAlive", "hasBomb", "nodeType", "areaId"]
        # print("Nodes data keys:", graph_dict["nodes_data"].values())

        # Extract graph features
        graph_data = graph_dict.get("graph_data", {})
        graph_features = [
            graph_data.get("seconds", 0) / 175.0,  # Normalize
        ]

        # Extract node features
        node_dicts = graph_dict["nodes_data"].values()
        node_features = []
        for node in node_dicts:
            # hp = node.get("hp", 0) / 100.0  # normalize
            # armor = node.get("armor", 0) / 100.0  # normalize
            utility = node.get("totalUtility", 0)

            norm_x = (node.get("x", 0) - self.global_min_x) / self.global_x_range
            norm_y = (node.get("y", 0) - self.global_min_y) / self.global_y_range

            area_onehot = self.area_encoder.transform([[node.get("areaId", 0)]])[0]

            binary_flags = [
                float(node.get("isAlive", 0)),
                float(node.get("hasBomb", 0)),
            ]

            full_feature = (
                [utility] + list(binary_flags) + list(area_onehot) + [norm_x, norm_y]
            )
            node_features.append(full_feature)

        x = torch.tensor(node_features, dtype=torch.float)

        # Create node index mapping
        node_ids = sorted(graph_dict["nodes_data"].keys())
        node_map = {nid: i for i, nid in enumerate(node_ids)}
        num_nodes = len(node_map)

        # Validate and build edge index
        edge_list = []
        for src, dst, _ in graph_dict["edges_data"]:
            if src in node_map and dst in node_map:
                if node_map[src] < num_nodes and node_map[dst] < num_nodes:
                    edge_list.append([node_map[src], node_map[dst]])
                else:
                    print(
                        f"Warning: Invalid edge {src}->{dst} in graph {graph_idx} from {file_path}"
                    )

        if not edge_list:
            raise ValueError(f"No valid edges in graph {graph_idx} from {file_path}")

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Add self-loops with validation
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # Extract label
        strategy = graph_dict.get("graph_data", {}).get("strategy_used", "unknown")
        label = self.label_to_id.get(strategy, 0)

        return Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(label, dtype=torch.long),
            graph_feat=torch.tensor(graph_features, dtype=torch.float),
        )


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, output_dim, graph_feat_dim=1):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels + graph_feat_dim, output_dim)
        self.dropout = Dropout(0.6)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)

        graph_feats = torch.stack([d.graph_feat for d in data.to_data_list()])
        x = torch.cat([x, graph_feats.to(x.device)], dim=1)

        return self.lin(x)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = "research_project/graphs"  # root directory of all game folders
    dataset = GraphDataset(data_path)

    # Print label distribution
    labels = [data.y.item() for data in dataset]
    label_counts = Counter(labels)
    print("Label distribution:", label_counts)

    # Compute class weights
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    # Split dataset
    train_len = int(0.8 * len(dataset))
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    print(f"Train set size: {len(train_set)} Test set size: {len(test_set)}")

    # Model setup
    sample_graph = dataset[0]
    graph_feat_dim = dataset[0].graph_feat.shape[0]
    input_dim = sample_graph.num_node_features
    output_dim = int(max(labels)) + 1
    hidden_channels = 64

    model = GNN(
        input_dim=input_dim,
        hidden_channels=hidden_channels,
        output_dim=output_dim,
        graph_feat_dim=graph_feat_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
        scheduler.step()

        train_acc = correct / total if total else 0
        print(
            f"Epoch {epoch}, Loss: {total_loss:.4f}, Training Accuracy: {train_acc:.2%}"
        )

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        num_classes = output_dim
        accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        precision = Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device)
        recall = Recall(task="multiclass", num_classes=num_classes, average="macro").to(
            device
        )
        f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(
            device
        )
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
                accuracy.update(pred, batch.y)
                precision.update(pred, batch.y)
                recall.update(pred, batch.y)
                f1.update(pred, batch.y)

        print(f"Accuracy: {accuracy.compute().item():.2%}")
        print(f"Precision: {precision.compute().item():.4f}")
        print(f"Recall: {recall.compute().item():.4f}")
        print(f"F1-score: {f1.compute().item():.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(dataset.area_encoder, "research_project/models/area_encoder.pkl")
    joblib.dump(dataset.label_to_id, "research_project/models/label_to_id.pkl")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hidden_channels": hidden_channels,
            "input_dim": input_dim,
            "output_dim": output_dim,
        },
        "research_project\models/checkpoint7_with_unknown.pt",
    )

    return model, dataset, class_weights


if __name__ == "__main__":
    pred_data = train()
