import joblib
import torch
from gnn import GNN, GraphDataset
from torch_geometric.loader import DataLoader


class Predictor:
    def __init__(self, model_path, dataset_path):
        self.model_path = model_path
        self.model_path = "research_project\models\checkpoint2.pt"
        self.dataset_path = dataset_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load unlabeled data
        print("Loading unlabeled data...")
        label_to_id = joblib.load("research_project/models/label_to_id2.pkl")
        self.id_to_label = {v: k for k, v in label_to_id.items()}
        dataset = GraphDataset(self.dataset_path, label_to_id=label_to_id)
        self.loader = DataLoader(dataset)

        print(f"Dataset loaded with {len(dataset)} graphs.")
        checkpoint = torch.load(self.model_path)
        output_dim = checkpoint["output_dim"]
        num_areas = checkpoint["num_areas"]

        self.model = GNN(
            hidden_channels=64, output_dim=output_dim, area_num_embeddings=num_areas
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def predict(self):
        predictions = []
        preds = []
        print("Predicting...")
        with torch.no_grad():
            for i, batch in enumerate(self.loader):
                batch = batch.to(self.device)
                out = self.model(batch)
                pred = out.argmax(dim=1)
                pred_labels = [self.id_to_label[p.item()] for p in pred]
                predictions.append(pred_labels)

        for i in predictions:
            preds.append(i[0])

        return preds
