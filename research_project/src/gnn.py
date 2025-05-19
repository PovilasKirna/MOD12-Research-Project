import os
import pickle
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Dataset, DataLoader, Data
from torch_geometric.nn import GCNConv, global_add_pool
from collections import Counter


class GraphDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.graph_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".pkl")]

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
                edge_attr.append([attr["dist"]])  # add more features if needed

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)


        label = raw_data["graph_data"].get("label", 0)
        y = torch.tensor([label], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)

# The Graph Neural Network Model
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch)
        return self.lin(x)

# Training function
def train():
    # Adjust this to your actual graph folder
    data_path = "csgo-analysis-main\graphs/00e7fec9-cee0-430f-80f4-6b50443ceacd"
    dataset = GraphDataset(data_path)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    print("Label distribution:", Counter([dataset.get(i).y.item() for i in range(len(dataset))]))

    sample_graph = dataset.get(0)
    input_dim = sample_graph.num_node_features
    output_dim = int(sample_graph.y.max().item() + 1)  # number of classes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(input_dim, hidden_dim=64, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 31):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")


if __name__ == "__main__":
    train()
 