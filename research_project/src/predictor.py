import joblib
import torch
from gnn import GNN, GraphDataset
from torch_geometric.loader import DataLoader


class Predictor:
    def __init__(self, model_path, dataset_path):
        self.model_path = model_path
        self.dataset_path = dataset_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load unlabeled data
        print("Loading unlabeled data...")
        area_encoder = joblib.load("research_project/models/area_encoder.pkl")
        label_to_id = joblib.load("research_project/models/label_to_id.pkl")
        self.id_to_label = {v: k for k, v in label_to_id.items()}
        dataset = GraphDataset(
            self.dataset_path, area_encoder=area_encoder, label_to_id=label_to_id
        )
        self.loader = DataLoader(dataset)

        print(f"Dataset loaded with {len(dataset)} graphs.")
        checkpoint = torch.load(model_path)
        input_dim = checkpoint["input_dim"]
        output_dim = checkpoint["output_dim"]

        self.model = GNN(input_dim=input_dim, hidden_channels=64, output_dim=output_dim)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def predict(self):
        predictions = []
        print("Predicting...")
        with torch.no_grad():
            for i, batch in enumerate(self.loader):
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                pred_labels = [self.id_to_label[p.item()] for p in pred]
                predictions.append(pred_labels)
                print(predictions[i])
        return predictions

'''
print("Starting prediction...")
Predictor(
    model_path="research_project\models\checkpoint1.pt",
    dataset_path="research_project\graphs\9aa73173-d219-4c36-9e49-6924ca12e2ed",
).predict()
'''