# %%
import os
import re
import sys
import csv
import math

from emb_models import *
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(parent_dir + '/SERGIO')
print(sys.path)

import yaml
import torch
import numpy as np
import torch.nn as nn
import seaborn as sns
from tqdm import tqdm
import networkx as nx
from torch import optim
from typing import List, Any
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.stats import ttest_ind
from sklearn.manifold import TSNE
from pytorch_lightning import Trainer
from experiment import GRNVAEExperiment
from dataset import GRNVAEDataset
from sklearn.decomposition import PCA
from scipy.sparse.linalg import bicgstab
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from scipy.spatial.distance import pdist, squareform
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.metrics import roc_auc_score, mean_squared_error, silhouette_score

from GENIE3.GENIE3 import *
nthreads=12

# %% [markdown]
# ### Get DS and Interactions info

# %%
def parse_dataset_name(folder_name):
    pattern1 = r'De-noised_(\d+)G_(\d+)T_(\d+)cPerT_dynamics_(\d+)_DS(\d+)'
    pattern2 = r'De-noised_(\d+)G_(\d+)T_(\d+)cPerT_(\d+)_DS(\d+)'
    match_p1 = re.match(pattern1, folder_name)
    match_p2 = re.match(pattern2, folder_name)
    if match_p1:
        return {
            'number_genes': int(match_p1.group(1)),
            'number_bins': int(match_p1.group(2)),
            'cells_per_type': int(match_p1.group(3)),
            'dynamics': int(match_p1.group(4)),
            'dataset_id': int(match_p1.group(5)),
            'folder_name': folder_name
        }
    if match_p2:
        return {
            'number_genes': int(match_p2.group(1)),
            'number_bins': int(match_p2.group(2)),
            'cells_per_type': int(match_p2.group(3)),
            'dynamics': int(match_p2.group(4)),
            'dataset_id': int(match_p2.group(5)),
            'folder_name': folder_name
        }
    return

def get_datasets():
    datasets = []
    data_sets_dir = '../SERGIO/data_sets'
    for folder_name in os.listdir(data_sets_dir):
        dataset_info = parse_dataset_name(folder_name)
        if dataset_info:
            datasets.append(dataset_info)
    # Include new datasets
    new_datasets = [
        {
            'dataset_id': 1001,
            'dataset_name': 'mESC',
            'expression_file': 'data/raws/mESC-ExpressionData.csv',
            'network_file': 'data/raws/mESC-network.csv',
        },
        {
            'dataset_id': 1002,
            'dataset_name': 'mHSC-E',
            'expression_file': 'data/raws/mHSC-E-ExpressionData.csv',
            'network_file': 'data/raws/mHSC-E-network.csv',
        },
        {
            'dataset_id': 1003,
            'dataset_name': 'mHSC-GM',
            'expression_file': 'data/raws/mHSC-GM-ExpressionData.csv',
            'network_file': 'data/raws/mHSC-GM-network.csv',
        },
        {
            'dataset_id': 1004,
            'dataset_name': 'mHSC-L',
            'expression_file': 'data/raws/mHSC-L-ExpressionData.csv',
            'network_file': 'data/raws/mHSC-L-network.csv',
        },
        {
            'dataset_id': 1005,
            'dataset_name': 'mESC-200',
            'expression_file': 'data/raws/mESC-200-ExpressionData.csv',
            'network_file': 'data/raws/mESC-200-network.csv',
        },
        {
            'dataset_id': 1006,
            'dataset_name': 'mHSC-E-200',
            'expression_file': 'data/raws/mHSC-E-200-ExpressionData.csv',
            'network_file': 'data/raws/mHSC-E-200-network.csv',
        },
    ]
    datasets.extend(new_datasets)
    return datasets

def load_data(dataset_info):
    dataset_id = dataset_info['dataset_id']
    if dataset_id < 1000:
        # Existing datasets
        data_dir = f'../SERGIO/imputation_data_2/DS{dataset_id}/'
        ds_clean_path = os.path.join(data_dir, 'DS6_clean.npy')
        ds_noisy_path = os.path.join(data_dir, 'DS6_45_iter_0.npy')
        if not os.path.exists(ds_clean_path) or not os.path.exists(ds_noisy_path):
            print(f"Data files not found for Dataset {dataset_id}. Skipping.")
            return None, None, None
        ds_clean = np.load(ds_clean_path).astype(np.float32)
        ds_noisy = np.load(ds_noisy_path).astype(np.float32)
        gene_names = None
    else:
        expression_file = dataset_info['expression_file']
        if not os.path.exists(expression_file):
            print(f"Expression file not found for Dataset {dataset_id}. Skipping.")
            return None, None, None
        import pandas as pd
        df = pd.read_csv(expression_file, index_col=0)
        ds_noisy = df.values.astype(np.float32)
        # Not available for new datasets, so set ds_noisy as raw data
        ds_clean = None  
        gene_names = df.index.tolist()
        # print("gene_names: ", gene_names)
    return ds_clean, ds_noisy, gene_names

def load_interactions_info(num_genes, interactions_file):
    gt = np.zeros((num_genes, num_genes))
    with open(interactions_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line_list = line.strip().split(',')
        target_index = int(float(line_list[0]))
        num_regs = int(float(line_list[1]))
        for i in range(num_regs):
            try:
                reg_index = int(float(line_list[i + 2]))
                gt[reg_index, target_index] = 1
            except:
                continue
    return gt

# %% [markdown]
# ### Get H and split into G where G = {train, valid} and H/G = {test}

# %%
def load_ground_truth_grn(file_path, num_genes=None, gene_names=None):
    if gene_names is None:
        # Existing datasets: indices
        if num_genes is None:
            with open(file_path, 'r') as f:
                indices = []
                for line in f:
                    source, target = map(int, line.strip().split(','))
                    indices.extend([source, target])
            num_genes = max(indices) + 1
        H = np.zeros((num_genes, num_genes))
        with open(file_path, 'r') as f:
            for line in f:
                source, target = map(int, line.strip().split(','))
                H[source, target] = 1
    else:
        # New datasets: gene names
        num_genes = len(gene_names)
        gene_to_idx = {gene.lower(): idx for idx, gene in enumerate(gene_names)}
        H = np.zeros((num_genes, num_genes))
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for line in reader:
                # print(line)
                source_gene, target_gene = line[0], line[1]
                if source_gene in gene_to_idx and target_gene in gene_to_idx:
                    source_idx = gene_to_idx[source_gene]
                    target_idx = gene_to_idx[target_gene]
                    H[source_idx, target_idx] = 1
    return H

def sample_partial_grn(H, sample_ratio=8/10):
    num_edges = np.sum(H)
    num_sample = int(num_edges * sample_ratio)
    edge_indices = np.argwhere(H == 1)
    np.random.shuffle(edge_indices)
    sampled_edges = edge_indices[:num_sample]
    G = np.zeros_like(H)
    G[tuple(zip(*sampled_edges))] = 1
    return G

def split_train_valid(G, train_ratio=7/8):  # 7:1 ratio
    edge_indices = np.argwhere(G == 1)
    num_edges = len(edge_indices)
    num_train = int(num_edges * train_ratio)
    np.random.shuffle(edge_indices)
    train_edges = edge_indices[:num_train]
    valid_edges = edge_indices[num_train:]
    G_train = np.zeros_like(G)
    G_valid = np.zeros_like(G)
    G_train[tuple(zip(*train_edges))] = 1
    G_valid[tuple(zip(*valid_edges))] = 1
    return G_train, G_valid

def get_test_set(H, G):
    G_test = H - G
    G_test[G_test < 0] = 0
    return G_test

# %% [markdown]
# ### Plot GRNs

# %%
def get_clusters_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    unclustered = set(G.nodes())
    cluster_labels = {}
    cluster_id = 0
    while unclustered:
        start_node = unclustered.pop()
        cluster = set()
        to_explore = {start_node}
        while to_explore:
            node = to_explore.pop()
            cluster.add(node)
            successors = set(G.successors(node)) - cluster
            to_explore.update(successors)
            predecessors = set(G.predecessors(node)) - cluster
            to_explore.update(predecessors)
        for node in cluster:
            cluster_labels[node] = cluster_id
            unclustered.discard(node)
        cluster_id += 1
    return cluster_labels

def get_cluster_labels(cluster_labels, num_genes):
    cluster_labels_list = np.full(num_genes, -1)
    for gene_idx in range(num_genes):
        if gene_idx in cluster_labels:
            cluster_labels_list[gene_idx] = cluster_labels[gene_idx]
    max_label = max(cluster_labels.values()) if cluster_labels else -1
    unassigned_label = max_label + 1
    cluster_labels_list[cluster_labels_list == -1] = unassigned_label
    return cluster_labels_list

def plot_grn_from_graphs(G_train, G_valid, G_test, H, dataset_id):
    num_genes = G_train.shape[0]
    cluster_labels_train = get_clusters_from_adj(G_train)
    cluster_labels_list_train = get_cluster_labels(cluster_labels_train, num_genes)
    cluster_labels_valid = get_clusters_from_adj(G_valid)
    cluster_labels_list_valid = get_cluster_labels(cluster_labels_valid, num_genes)
    cluster_labels_test = get_clusters_from_adj(G_test)
    cluster_labels_list_test = get_cluster_labels(cluster_labels_test, num_genes)
    cluster_labels_H = get_clusters_from_adj(H)
    cluster_labels_list_H = get_cluster_labels(cluster_labels_H, num_genes)

    plot_grn(G_train, cluster_labels_list_train, 'Training Set GRN', f'./results/cl/DS{dataset_id}/grn_train.png')
    plot_grn(G_valid, cluster_labels_list_valid, 'Validation Set GRN', f'./results/cl/DS{dataset_id}/grn_valid.png')
    plot_grn(G_test, cluster_labels_list_test, 'Test Set GRN', f'./results/cl/DS{dataset_id}/grn_test.png')
    plot_grn(H, cluster_labels_list_H, 'Full Ground Truth GRN', f'./results/cl/DS{dataset_id}/grn_full.png')

    return cluster_labels_list_train, cluster_labels_list_valid, cluster_labels_list_test, cluster_labels_list_H

def plot_grn(adj_matrix, cluster_labels, title, save_path):
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=42)
    node_colors = cluster_labels[list(G.nodes())]
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_size=50,
                           cmap=plt.cm.get_cmap('nipy_spectral', int(np.max(node_colors)) + 1),
                           node_color=node_colors)
    nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True)
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def plot_embeddings(embeddings, cluster_labels, title, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          cmap=plt.cm.get_cmap('nipy_spectral', int(np.max(cluster_labels))+1),
                          c=cluster_labels, s=50, alpha=0.7)
    plt.colorbar(scatter, label='Cluster Labels')
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

# %% [markdown]
# # Contrastive Learning Algo

# %%
def get_balanced_edges(adj_matrix, num_batches):
    positive_edges = np.argwhere(adj_matrix == 1)
    negative_edges = np.argwhere(adj_matrix == 0)
    num_positive = len(positive_edges)
    num_negative = len(negative_edges)
    edges_per_batch = num_positive // num_batches
    
    np.random.shuffle(positive_edges)
    np.random.shuffle(negative_edges)
    
    balanced_edges = []
    for i in range(num_batches):
        batch_positive = positive_edges[i * edges_per_batch : (i + 1) * edges_per_batch]
        batch_negative = negative_edges[i * edges_per_batch : (i + 1) * edges_per_batch]
        balanced_edges.append(np.concatenate([batch_positive, batch_negative]))
    
    return balanced_edges

def edge_score(embedding_i, embedding_j):
    # Element-wise multiplication followed by summation
    print(f"shape of embedding_i: {embedding_i.shape}, embedding_j: {embedding_j.shape}")
    scores = torch.sum(embedding_i * embedding_j, dim=1)  # Shape: [batch_size]
    print(f"shape of scores: {scores.shape}")
    print(f"shape of torch.sigmoid(scores): {torch.sigmoid(scores).shape}")
    return torch.sigmoid(scores)  # Shape: [batch_size]

def compute_edge_likelihoods(embeddings):
    similarity_matrix = np.dot(embeddings, embeddings.T)
    probabilities = 1 / (1 + np.exp(-similarity_matrix))  # Sigmoid to map to [0, 1]
    return probabilities

from pathlib import Path

def VAE_embeddings(ds, G_train, G_valid, dataset_id, config_path):
    with open(config_path, 'r') as file:
        try:   
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=f"{config['model_params']['name']}_DS{dataset_id}")

    seed_everything(config['exp_params']['manual_seed'], True)
    print(f"ds.shape: {ds} {ds.shape[0]} {ds.shape[1]}")
    print(f"ds: {ds}")
    # model = VanillaVAE(num_genes=ds.shape[0], num_cells=ds.shape[1], **config['model_params'])
    model = BetaVAE(num_genes=ds.shape[0], num_cells=ds.shape[1], batch_size=config['data_params']['train_batch_size'], **config['model_params'])
    num_genes = ds.shape[0]
    num_cells = ds.shape[1]
    other_params = {
        'dataset_id': dataset_id,
        'num_genes': num_genes,
        'num_cells': num_cells,
    }
    experiment = GRNVAEExperiment(model, {**config['exp_params'], **config['model_params'], **config['data_params'], **config['trainer_params'], **other_params})

    data = GRNVAEDataset(
        data=ds,
        adjacency_matrix=G_train,  # Use G_train as the adjacency matrix
        **config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0
    )
    data.setup()

    runner = Trainer(logger=tb_logger,
                    callbacks=[
                        LearningRateMonitor(),
                        ModelCheckpoint(save_top_k=2, 
                                        dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                        monitor= "valid total loss",
                                        save_last= True),
                    ],
                    strategy=DDPStrategy(find_unused_parameters=False),
                    **config['trainer_params'])

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training VAE for Dataset {dataset_id} =======")
    runner.fit(experiment, datamodule=data)

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(torch.tensor(ds, dtype=torch.float))[0]
        embeddings = embeddings.cpu().numpy()

    return embeddings

def PCA_embeddings(ds, n_components=64):
    pca = PCA(n_components=n_components, random_state=42)
    print("ds.shape: ", ds.shape)
    embeddings = pca.fit_transform(ds)  # Shape: (num_genes, n_components)
    return embeddings

# %%
datasets = get_datasets()
for dataset_info in datasets[:1]:
    dataset_id = dataset_info['dataset_id']
    print(f"\nProcessing Dataset {dataset_id}...")
    ds_clean, ds_noisy, gene_names = load_data(dataset_info)
    if ds_noisy is None:
        continue

    num_genes = ds_noisy.shape[0]
    if dataset_id < 1000:
        # simulated data
        gt_grn_file = f'../SERGIO/data_sets/{dataset_info["folder_name"]}/gt_GRN.csv'
        H = load_ground_truth_grn(gt_grn_file)
    else:
        # real data
        # print(f"gene_names: {gene_names}")
        gt_grn_file = dataset_info['network_file']
        H = load_ground_truth_grn(gt_grn_file, gene_names=gene_names)

    G = sample_partial_grn(H, sample_ratio=0.9)
    G_train, G_valid = split_train_valid(G, train_ratio=8/9)  # 8:1 ratio
    G_test = get_test_set(H, G)

    os.makedirs(f'./results/cl/DS{dataset_id}/', exist_ok=True)
    # PCA embeddings
    # embeddings = PCA_embeddings(ds_noisy, n_components=64)
    # np.save(f'./results/cl/DS{dataset_id}/pca_embeddings.npy', embeddings)
    # print(f"PCA embeddings saved for Dataset {dataset_id} at ./results/cl/DS{dataset_id}/pca_embeddings.npy")
    # VAE embeddings
    embeddings = VAE_embeddings(ds_noisy, G_train, G_valid, dataset_id, config_path='./configs/bbvae.yaml')
    np.save(f'./results/cl/DS{dataset_id}/vae_embeddings.npy', embeddings)
    print(f"VAE embeddings saved for Dataset {dataset_id} at ./results/cl/DS{dataset_id}/vae_embeddings.npy")

    # ds_clean: 1200*2700
    # ds_noisy: 1200*2700
    # G_train: 1200*1200
    # G_valid: 1200*1200
    # G_test: 1200*1200
    # H: 1200*1200
    # embeddings: 1200*64 ngenes*latent_dim
    # print(f"shapes of datasets: {ds_clean.shape} {ds_noisy.shape} {G_train.shape}, {G_valid.shape}, {G_test.shape}, {H.shape} {embeddings.shape}")
