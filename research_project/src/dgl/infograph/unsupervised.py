import argparse
from pathlib import Path

import dgl
import numpy as np
import torch as th
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GNNExplainer
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

import stats
from evaluate_embedding import evaluate_embedding
from model import InfoGraph
from estaDataset import EstaDataset
import matplotlib.pyplot as plt


def argument():
    parser = argparse.ArgumentParser(description="InfoGraph")
    # data source params
    parser.add_argument(
        "--dataname", type=str, default="MUTAG", help="Name of dataset."
    )

    # training params
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU index, default:-1, using CPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Training batch size."
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="Interval between two evaluations.",
    )

    # model params
    parser.add_argument(
        "--n_layers",
        type=int,
        default=3,
        help="Number of graph convolution layers before each pooling.",
    )
    parser.add_argument(
        "--hid_dim", type=int, default=32, help="Hidden layer dimensionalities."
    )

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and th.cuda.is_available():
        args.device = "cuda:{}".format(args.gpu)
    else:
        args.device = "cpu"

    return args


def collate(samples):
    """collate function for building graph dataloader"""

    graphs, labels = map(list, zip(*samples))

    # generate batched graphs and labels
    batched_graph = dgl.batch(graphs)
    batched_labels = th.tensor(labels)

    n_graphs = len(graphs)
    graph_id = th.arange(n_graphs)
    graph_id = dgl.broadcast_nodes(batched_graph, graph_id)

    batched_graph.ndata["graph_id"] = graph_id

    return batched_graph, batched_labels


if __name__ == "__main__":
    # Step 1: Prepare graph data   ===================================== #
    args = argument()
    print(args)

    # load dataset from dgl.data.GINDataset
    dataset = GINDataset(args.dataname, self_loop=False, force_reload=True)
    # data_folder = Path(__file__).parent / "../../../graphs/" / stats.EXAMPLE_DEMO_PATH.stem
    # data_folder = str(data_folder.resolve())
    # dataset = EstaDataset(raw_dir=data_folder)
    print("data set loaded")

    # get graphs and labels
    graphs, labels = map(list, zip(*dataset))
    print("graphs, labels mapped")

    # generate a full-graph with all examples for evaluation
    wholegraph = dgl.batch(graphs)
    print("wholegraph = dgl.batch(graphs)")

    wholegraph.ndata["attr"] = wholegraph.ndata["attr"].to(th.float32)
    print("th.float32")

    # create dataloader for batch training
    dataloader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate,
        drop_last=False,
        shuffle=True,
    )

    in_dim = wholegraph.ndata["attr"].shape[1]

    ### **TODO** edge featuers in InfoGraph with GINEConv
    ### TODO: what value is needed for in_dim?

    # Step 2: Create model =================================================================== #
    model = InfoGraph(in_dim, args.hid_dim, args.n_layers)
    model = model.to(args.device)

    # Step 3: Create training components ===================================================== #
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    print("===== Before training ======")

    wholegraph = wholegraph.to(args.device)
    wholefeat = wholegraph.ndata["attr"]

    emb = model.get_embedding(wholegraph, whole_nfeat, whole_efeat).cpu()
    # res = evaluate_embedding(emb, labels, args.device)

    """ Evaluate the initialized embeddings """
    """ using logistic regression and SVM(non-linear) """
    # print("logreg {:4f}, svc {:4f}".format(res[0], res[1]))

    best_logreg = 0
    best_logreg_epoch = 0
    best_svc = 0
    best_svc_epoch = 0

    # Step 4: training epochs =============================================================== #
    for epoch in range(100): #args.epochs):
        loss_all = 0
        model.train()

        for graph, label in dataloader:
            graph = graph.to(args.device)
            feat = graph.ndata["attr"]
            graph_id = graph.ndata["graph_id"]

            n_graph = label.shape[0]

            optimizer.zero_grad()
            loss = model(graph, feat, graph_id)
            loss.backward()
            optimizer.step()
            loss_all += loss.item()

        print("Epoch {}, Loss {:.4f}".format(epoch, loss_all))

        if epoch % args.log_interval == 0:
            # evaluate embeddings
            model.eval()
            emb = model.get_embedding(wholegraph, wholefeat).cpu()
            # res = evaluate_embedding(emb, labels, args.device)

            # if res[0] > best_logreg:
            #     best_logreg = res[0]
            #     best_logreg_epoch = epoch
            #
            # if res[1] > best_svc:
            #     best_svc = res[1]
            #     best_svc_epoch = epoch
            #
            # print(
            #     "best logreg {:4f}, epoch {} | best svc: {:4f}, epoch {}".format(
            #         best_logreg, best_logreg_epoch, best_svc, best_svc_epoch
            #     )
            # )

    print("Training End")
    print("best logreg {:4f} ,best svc {:4f}".format(best_logreg, best_svc))

    # eval mode and get embedding for whole graph
    model.eval()
    emb = model.get_embedding(wholegraph, wholefeat).cpu()

    # create t-sne for visualization
    embedding_array = np.array(emb)
    X_embedded = TSNE(n_components=2).fit_transform(embedding_array)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.show()

    # feature explainer TODO doesnt currently work with model
    # explainer = GNNExplainer(model, num_hops=1)
    # graph = dataset[0][0]
    # feat = graph.ndata["attr"]
    # graph_id = 0
    # feat_mask, edge_mask = explainer.explain_graph(wholegraph, wholefeat)
    # print(feat_mask)

    # dbscan
    db = DBSCAN(eps=15, min_samples=5).fit(embedding_array)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = embedding_array[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = embedding_array[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()