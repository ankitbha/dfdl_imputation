# %%
import os
import re
import sys
import csv
import json
import torch
import queue
import math
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
import torch.optim as optim
from itertools import product
import matplotlib.pyplot as plt
import torch.nn.functional as F
import datetime
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    confusion_matrix,
    matthews_corrcoef
)
from lightning_lite.utilities.seed import seed_everything

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

def load_network_data(file_path, gene_list):
    # Map gene names to indices
    gene_to_index = {gene: idx for idx, gene in enumerate(gene_list)}
    num_genes = len(gene_list)
    H = np.zeros((num_genes, num_genes))
    import csv
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            source_gene, target_gene = line
            if source_gene in gene_to_index and target_gene in gene_to_index:
                source_idx = gene_to_index[source_gene]
                target_idx = gene_to_index[target_gene]
                H[source_idx, target_idx] = 1
    return H

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

def identify_tfs(H):
    out_degrees = np.sum(H, axis=1)
    in_degrees = np.sum(H, axis=0)
    tf_indices = np.where((out_degrees > in_degrees) & (out_degrees > 0))[0]
    return tf_indices

# %%
def load_ground_truth_grn(file_path, num_genes=None, gene_names=None):
    if gene_names is None:
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
        num_genes = len(gene_names)
        gene_to_idx = {gene.lower(): idx for idx, gene in enumerate(gene_names)}
        H = np.zeros((num_genes, num_genes))
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for line in reader:
                source_gene, target_gene = line[0].lower(), line[1].lower()
                if source_gene in gene_to_idx and target_gene in gene_to_idx:
                    source_idx = gene_to_idx[source_gene]
                    target_idx = gene_to_idx[target_gene]
                    H[source_idx, target_idx] = 1
    return H

def sample_partial_grn(H, sample_ratio=8/10, cv_split_axis=0):
    tf_indices = identify_tfs(H)
    if cv_split_axis == 0:
        tf_perm = np.random.permutation(len(tf_indices))
        split_idx = int(len(tf_perm) * sample_ratio)
        sampled_tfs = tf_indices[tf_perm[:split_idx]]
        G = -np.ones_like(H)
        G[sampled_tfs, :] = H[sampled_tfs, :]
    else:
        gene_perm = np.random.permutation(H.shape[1])
        split_idx = int(len(gene_perm) * sample_ratio)
        sampled_genes = gene_perm[:split_idx]
        G = -np.ones_like(H)
        G[tf_indices[:, None], sampled_genes] = H[tf_indices[:, None], sampled_genes]
    return G

def split_train_valid(G, train_ratio=7/8, cv_split_axis=0):
    tf_indices = identify_tfs(G)
    if cv_split_axis == 0:
        known_tfs = tf_indices[~np.all(G[tf_indices] == -1, axis=1)]
        perm = np.random.permutation(len(known_tfs))
        split_idx = int(len(perm) * train_ratio)
        train_tfs = known_tfs[perm[:split_idx]]
        valid_tfs = known_tfs[perm[split_idx:]]
        G_train = -np.ones_like(G)
        G_valid = -np.ones_like(G)
        G_train[train_tfs, :] = G[train_tfs, :]
        G_valid[valid_tfs, :] = G[valid_tfs, :]
    else:
        known_genes = np.where(~np.all(G[tf_indices[:, None], :] == -1, axis=0))[0]
        perm = np.random.permutation(len(known_genes))
        split_idx = int(len(perm) * train_ratio)
        train_genes = known_genes[perm[:split_idx]]
        valid_genes = known_genes[perm[split_idx:]]
        G_train = -np.ones_like(G)
        G_valid = -np.ones_like(G)
        G_train[tf_indices[:, None], train_genes] = G[tf_indices[:, None], train_genes]
        G_valid[tf_indices[:, None], valid_genes] = G[tf_indices[:, None], valid_genes]
    return G_train, G_valid


def get_test_set(H, G, cv_split_axis=0):
    tf_indices = identify_tfs(H)
    G_test = -np.ones_like(H)
    if cv_split_axis == 0:
        test_tfs = tf_indices[np.all(G[tf_indices] == -1, axis=1)]
        G_test[test_tfs, :] = H[test_tfs, :]
    else:
        test_genes = np.where(np.all(G[tf_indices[:, None], :] == -1, axis=0))[0]
        G_test[tf_indices[:, None], test_genes] = H[tf_indices[:, None], test_genes]
    return G_test
# %%
def generate_balanced_batches(embeddings, adjacency_matrix, batch_size, num_batches):
    num_nodes = embeddings.shape[0]
    positive_edges = np.argwhere(adjacency_matrix == 1)
    negative_edges = np.argwhere(adjacency_matrix == 0)
    
    np.random.shuffle(positive_edges)
    np.random.shuffle(negative_edges)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(positive_edges))
        
        batch_positive = positive_edges[start_idx:end_idx]
        batch_negative = negative_edges[np.random.choice(len(negative_edges), size=len(batch_positive), replace=False)]

        batch_edges = np.concatenate([batch_positive, batch_negative])
        batch_labels = np.concatenate([np.ones(len(batch_positive)), np.zeros(len(batch_negative))])
        
        # Shuffle the batch
        shuffle_idx = np.random.permutation(len(batch_edges))
        batch_edges = batch_edges[shuffle_idx]
        batch_labels = batch_labels[shuffle_idx]
        
        x1 = torch.tensor(embeddings[batch_edges[:, 0]], dtype=torch.float32).to(device)
        x2 = torch.tensor(embeddings[batch_edges[:, 1]], dtype=torch.float32).to(device)
        labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
        yield x1, x2, labels

def generate_batches(embeddings, adjacency_matrix, batch_size, num_batches, negative_ratio=10):
    tf_indices = identify_tfs(adjacency_matrix)
    positive_edges = np.argwhere((adjacency_matrix == 1) & (np.isin(np.arange(adjacency_matrix.shape[0])[:, None], tf_indices)))
    negative_edges = np.argwhere((adjacency_matrix == 0) & (np.isin(np.arange(adjacency_matrix.shape[0])[:, None], tf_indices)))
    np.random.shuffle(positive_edges)
    np.random.shuffle(negative_edges)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(positive_edges))
        batch_positive = positive_edges[start_idx:end_idx]
        num_negatives = len(batch_positive) * negative_ratio
        batch_negative = negative_edges[np.random.choice(len(negative_edges), size=num_negatives, replace=len(negative_edges) < num_negatives)]
        batch_edges = np.concatenate([batch_positive, batch_negative])
        batch_labels = np.concatenate([np.ones(len(batch_positive)), np.zeros(len(batch_negative))])
        shuffle_idx = np.random.permutation(len(batch_edges))
        batch_edges = batch_edges[shuffle_idx]
        batch_labels = batch_labels[shuffle_idx]
        x1 = torch.tensor(embeddings[batch_edges[:, 0]], dtype=torch.float32)
        x2 = torch.tensor(embeddings[batch_edges[:, 1]], dtype=torch.float32)
        labels = torch.tensor(batch_labels, dtype=torch.float32)
        yield x1, x2, labels
# %%
class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, projection_dim):
        super(ContrastiveModel, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
        # self.projection = nn.Sequential(
        #     nn.Linear(input_dim, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, projection_dim)
        # )
    
    def forward(self, x1, x2):
        proj1 = self.projection(x1)
        proj2 = self.projection(x2)
        return proj1, proj2

# %%
# class SoftNearestNeighborLoss(nn.Module):
#     def __init__(self, temperature=0.5):
#         super(SoftNearestNeighborLoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, embeddings, labels):
#         """
#         embeddings: Tensor of shape [batch_size * 2, projection_dim]
#         labels: Tensor of shape [batch_size * 2]
#         """
#         batch_size = embeddings.size(0)
#         embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
#         sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature  # [batch_size*2, batch_size*2]
#         if torch.isnan(sim_matrix).any():
#             print("NaN detected in sim_matrix")
#         mask = torch.eye(batch_size, device=embeddings.device).bool()
#         if torch.isnan(mask).any():
#             print("NaN detected in mask")
#         sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
#         if torch.isnan(sim_matrix).any():
#             print("NaN detected in sim_matrix after masking")
#         sim_exp = torch.exp(sim_matrix)
#         if torch.isnan(sim_exp).any():
#             print("NaN detected in sim_exp")
#         sim_exp_sum = sim_exp.sum(dim=1, keepdim=True)
#         if torch.isnan(sim_exp_sum).any():
#             print("NaN detected in sim_exp_sum")
#         # print(sim_matrix)
#         sim_probs = sim_exp / (sim_exp_sum + 1)
#         if torch.isnan(sim_probs).any():
#             print("NaN detected in sim_probs")
#         labels = labels.unsqueeze(1)
#         if torch.isnan(labels).any():
#             print("NaN detected in labels")
#         label_equal = labels == labels.T
#         if torch.isnan(label_equal).any():
#             print("NaN detected in label_equal")
#         loss = -torch.log((sim_probs * label_equal.float()).sum(dim=1) + 1e-8)
#         if torch.isnan(loss).any():
#             print("NaN detected in loss")
#         return loss.mean()

# def train_snn(embeddings, adjacency_matrix, input_dim, projection_dim, num_epochs, batch_size, learning_rate, device):
#     model = ContrastiveModel(input_dim, projection_dim).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = SoftNearestNeighborLoss(temperature=0.5)
    
#     num_positive_edges = len(np.argwhere(adjacency_matrix == 1))
#     num_batches = num_positive_edges // batch_size
#     print("num_positive_edges: ", num_positive_edges, num_batches)
    
#     for epoch in range(num_epochs):
#         total_loss = 0
#         model.train()
#         for x1, x2, labels in generate_balanced_batches(embeddings, adjacency_matrix, batch_size, num_batches):
#             x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
#             optimizer.zero_grad()
#             proj1, proj2 = model(x1, x2)
#             # combine projections and labels for loss computation
#             embeddings_batch = torch.cat([proj1, proj2], dim=0)
#             labels_batch = torch.cat([labels, labels], dim=0)
#             loss = criterion(embeddings_batch, labels_batch)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
        
#         avg_loss = total_loss / num_batches
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
#     return model

class SoftNearestNeighborLoss(nn.Module):
    def __init__(self, temperature=10., cos_distance=True):
        super(SoftNearestNeighborLoss, self).__init__()
        self.temperature = temperature
        self.cos_distance = cos_distance

    def pairwise_cos_distance(self, A, B):
        query_embeddings = F.normalize(A, dim=1)
        key_embeddings = F.normalize(B, dim=1)
        distances = 1 - torch.matmul(query_embeddings, key_embeddings.T)
        return distances

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Batched embeddings to compute the SNNL.
            labels: Labels of embeddings.
        """
        batch_size = embeddings.shape[0]
        eps = 1e-9

        if self.cos_distance:
            pairwise_dist = self.pairwise_cos_distance(embeddings, embeddings)
        else:
            pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        pairwise_dist = pairwise_dist / self.temperature
        negexpd = torch.exp(-pairwise_dist)

        # Creating mask to sample same class neighborhood
        pairs_y = torch.broadcast_to(labels, (batch_size, batch_size))
        mask = pairs_y == torch.transpose(pairs_y, 0, 1)
        mask = mask.float()

        # creating mask to exclude diagonal elements
        ones = torch.ones([batch_size, batch_size], dtype=torch.float32).cuda()
        dmask = ones - torch.eye(batch_size, dtype=torch.float32).cuda()

        # all class neighborhood
        alcn = torch.sum(torch.multiply(negexpd, dmask), dim=1)
        # same class neighborhood
        sacn = torch.sum(torch.multiply(negexpd, mask), dim=1)

        # Adding eps for numerical stability
        loss = -torch.log((sacn+eps)/alcn).mean()
        return loss

def train_snn(embeddings, adjacency_matrix, input_dim, projection_dim, num_epochs, batch_size, learning_rate, negative_ratio, temperature, device):
    # print("0")
    model = ContrastiveModel(input_dim, projection_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = SoftNearestNeighborLoss(temperature=temperature)
    num_positive_edges = (adjacency_matrix == 1).sum().item()
    num_batches = num_positive_edges // batch_size
    print(f"num_positive_edges: {num_positive_edges}, num_batches: {num_batches}")

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for x1, x2, labels in generate_batches(embeddings, adjacency_matrix, batch_size, num_batches, negative_ratio=negative_ratio):
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            optimizer.zero_grad()
            proj1, proj2 = model(x1, x2)
            embeddings_batch = torch.cat([proj1, proj2], dim=0)
            labels_batch = torch.cat([labels, labels], dim=0)
            loss = criterion(embeddings_batch, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return model

def train_cl(embeddings, adjacency_matrix, input_dim, projection_dim, num_epochs, batch_size, learning_rate, negative_ratio, device):
    model = ContrastiveModel(input_dim, projection_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    margin = 1.0
    num_positive_edges = len(np.argwhere(adjacency_matrix == 1))
    num_batches = num_positive_edges // batch_size
    # print("num_positive_edges: ", num_positive_edges, num_batches)
    
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for x1, x2, labels in generate_batches(embeddings, adjacency_matrix, batch_size, num_batches, negative_ratio=negative_ratio):
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device).float()
            optimizer.zero_grad()
            proj1, proj2 = model(x1, x2)
            distances = F.pairwise_distance(proj1, proj2)
            loss = torch.mean(labels * distances.pow(2) + (1 - labels) * F.relu(margin - distances).pow(2))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

def train_bce(embeddings, adjacency_matrix, input_dim, projection_dim, num_epochs, batch_size, learning_rate, negative_ratio, device):
    model = ContrastiveModel(input_dim, projection_dim).to(device)
    num_positive = np.sum(adjacency_matrix == 1)
    num_negative = np.sum(adjacency_matrix == 0)
    pos_weight = torch.tensor(num_negative / num_positive).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_positive_edges = len(np.argwhere(adjacency_matrix == 1))
    num_batches = num_positive_edges // batch_size
    # print("num_positive_edges: ", num_positive_edges, num_batches)
    
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for x1, x2, labels in generate_batches(embeddings, adjacency_matrix, batch_size, num_batches, negative_ratio=negative_ratio):
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device).float()
            optimizer.zero_grad()
            proj1, proj2 = model(x1, x2)
            # proj1 = nn.functional.normalize(proj1, p=2, dim=1)
            # proj2 = nn.functional.normalize(proj2, p=2, dim=1)
            sim_scores = torch.sum(proj1 * proj2, dim=1)
            loss = criterion(sim_scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

def train_cel(embeddings, adjacency_matrix, input_dim, projection_dim, num_epochs, batch_size, learning_rate, negative_ratio, device):
    model = ContrastiveModel(input_dim, projection_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CosineEmbeddingLoss(margin=0.0)
    
    num_positive_edges = len(np.argwhere(adjacency_matrix == 1))
    num_batches = num_positive_edges // batch_size
    # print("num_positive_edges: ", num_positive_edges, num_batches)
    
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for x1, x2, labels in generate_batches(embeddings, adjacency_matrix, batch_size, num_batches, negative_ratio=negative_ratio):
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device).float()
            cosine_labels = labels * 2 - 1
            optimizer.zero_grad()
            proj1, proj2 = model(x1, x2)
            loss = criterion(proj1, proj2, cosine_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model


# %%
def get_k_metrics(true_labels, predicted_scores):
    k = int(np.sum(true_labels))
    sorted_indices = np.argsort(predicted_scores)[::-1]
    sorted_labels = true_labels[sorted_indices]
    precision_k = np.sum(sorted_labels[:k]) / k
    recall_k = np.sum(sorted_labels[:k]) / np.sum(true_labels)
    return precision_k, recall_k

def format_metric_with_std(name, metrics, metric_key):
    mean = metrics['means'][metric_key]
    std = metrics['stds'][metric_key]
    ci = metrics['ci'][metric_key]
    return f"{name}: {mean:.4f} +- {std:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])"

def print_evaluation_results(metrics):
    print("Evaluation Results:")
    print("=" * 50)
    print(format_metric_with_std("AUC-ROC", metrics, 'auc_roc'))
    print(format_metric_with_std("AUC-PR", metrics, 'auc_pr'))
    print(format_metric_with_std("F1 Score", metrics, 'f1_score'))
    print(format_metric_with_std("MCC", metrics, 'mcc'))
    print(format_metric_with_std("K-Precision", metrics, 'k_precision'))
    print(format_metric_with_std("K-Recall", metrics, 'k_recall'))

def evaluate_contrastive_model(model, embeddings, adjacency_matrix, loss_function, device):
    """Evaluation using full test set to maintain consistency"""
    # print("1")
    model.eval()
    with torch.no_grad():
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
        projected_embeddings = model.projection(embeddings_tensor)
        projected_embeddings = nn.functional.normalize(projected_embeddings, p=2, dim=1)
        projected_embeddings = projected_embeddings.cpu().numpy()
        # print("2")
        # Get all edges in test set
        test_edges = np.argwhere(adjacency_matrix != -1)  # -1 for ignored edges
        true_labels = adjacency_matrix[test_edges[:, 0], test_edges[:, 1]]
        # print("3")
        # Compute scores for all edges
        emb1 = projected_embeddings[test_edges[:, 0]]
        emb2 = projected_embeddings[test_edges[:, 1]]
        similarity_scores = np.sum(emb1 * emb2, axis=1)
        # print("4")
        if loss_function == 'SNN':
            similarity_scores = 1 - similarity_scores
        probabilities = 1 / (1 + np.exp(-similarity_scores))
        # print("5")
        print(f"Evaluating on {len(true_labels)} edges")
        print(f"Positive edges: {np.sum(true_labels == 1)}")
        print(f"Negative edges: {np.sum(true_labels == 0)}")
        
        metrics = {}
        print("true_labels: ", true_labels)
        print("true_labels.shape: ", true_labels.shape)
        print("probabilities: ", probabilities)
        print("probabilities.shape: ", probabilities.shape)
        metrics['auc_roc'] = roc_auc_score(true_labels, probabilities)
        precision, recall, thresholds = precision_recall_curve(true_labels, probabilities)
        metrics['auc_pr'] = auc(recall, precision)
        y_pred = (probabilities >= 0.5).astype(int)
        print("y_pred: ", y_pred)
        print("y_pred.shape: ", y_pred.shape)
        metrics['f1_score'] = f1_score(true_labels, y_pred, zero_division=0)
        metrics['confusion_matrix'] = confusion_matrix(true_labels, y_pred)
        metrics['mcc'] = matthews_corrcoef(true_labels, y_pred)
        k_precision, k_recall = get_k_metrics(true_labels, probabilities)
        metrics['k_precision'] = k_precision
        metrics['k_recall'] = k_recall
        
        # Store all data
        metrics.update({
            'precision_curve': precision,
            'recall_curve': recall,
            'thresholds': thresholds,
            'true_labels': true_labels,
            'probabilities': probabilities,
            'num_positive': np.sum(true_labels == 1),
            'num_negative': np.sum(true_labels == 0)
        })
        
        return metrics

# %%
def plot_distribution(scores, labels, set_name, log_dir):
        plt.figure(figsize=(12, 6))
        probabilities = 1 / (1 + np.exp(-scores))
        sns.kdeplot(probabilities[labels == 0], fill=True, color="skyblue", label="Negative", cut=0)
        sns.kdeplot(probabilities[labels == 1], fill=True, color="red", label="Positive", cut=0)
        
        plt.title(f'{set_name} Set Predicted Probabilities Distribution')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.legend()
        total = len(labels)
        pos_prop = np.sum(labels == 1) / total
        neg_prop = 1 - pos_prop
        plt.text(0.05, 0.95, f"Negative: {neg_prop:.2%}\nPositive: {pos_prop:.2%}", 
                transform=plt.gca().transAxes, verticalalignment='top')
        plt.savefig(os.path.join(log_dir, f'{set_name.lower()}_distribution.png'))
        plt.close()

def log_experiment(result):
    dataset_id = result['dataset_id']
    embedding_type = result['embedding_type']
    loss_function = result['loss_function']

    base_dir = f'./logs/{embedding_type}_{loss_function}_DS{dataset_id}/'
    os.makedirs(base_dir, exist_ok=True)
    existing_versions = [int(d.split('_')[1]) for d in os.listdir(base_dir) if d.startswith('version_')]
    next_version = max(existing_versions + [0]) + 1
    
    log_dir = os.path.join(base_dir, f'version_{next_version}')
    os.makedirs(log_dir, exist_ok=True)
    
    # Save parameters and results
    params = {k: v for k, v in result.items() if k not in ['model_state_dict', 'projected_embeddings', 'train_labels', 'train_scores', 'valid_labels', 'valid_scores', 'test_labels', 'test_scores']}
    
    with open(os.path.join(log_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)
    
    # Generate and save plots
    plot_distribution(result['train_scores'], result['train_labels'], 'Train', log_dir)
    plot_distribution(result['valid_scores'], result['valid_labels'], 'Validation', log_dir)
    plot_distribution(result['test_scores'], result['test_labels'], 'Test', log_dir)
    
    # Save model and embeddings
    torch.save(result['model_state_dict'], os.path.join(log_dir, f'cl_model_{embedding_type}_{loss_function}.pth'))
    np.save(os.path.join(log_dir, f'projected_embeddings_{embedding_type}_{loss_function}.npy'), result['projected_embeddings'])
    
    return log_dir

def save_split_data(dataset_id, split_name, expression_data, network_data, base_dir='./data/splits'):
    dataset_dir = os.path.join(base_dir, f'DS{dataset_id}')
    split_dir = os.path.join(dataset_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    genes = [f'Gene_{i}' for i in range(expression_data.shape[0])]
    cells = [f'Cell_{i}' for i in range(expression_data.shape[1])]
    expression_df = pd.DataFrame(expression_data, index=genes, columns=cells)
    expression_df.to_csv(os.path.join(split_dir, 'ExpressionData.csv'))

    positive_edges = np.argwhere(network_data == 1)
    negative_edges = np.argwhere(network_data == 0)
    
    pos_edge_df = pd.DataFrame(positive_edges, columns=['Gene1', 'Gene2'])
    pos_edge_df['Gene1'] = pos_edge_df['Gene1'].apply(lambda x: f'Gene_{x}')
    pos_edge_df['Gene2'] = pos_edge_df['Gene2'].apply(lambda x: f'Gene_{x}')
    pos_edge_df.to_csv(os.path.join(split_dir, 'pos_refNetwork.csv'), index=False)
    
    neg_edge_df = pd.DataFrame(negative_edges, columns=['Gene1', 'Gene2'])
    neg_edge_df['Gene1'] = neg_edge_df['Gene1'].apply(lambda x: f'Gene_{x}')
    neg_edge_df['Gene2'] = neg_edge_df['Gene2'].apply(lambda x: f'Gene_{x}')
    neg_edge_df.to_csv(os.path.join(split_dir, 'neg_refNetwork.csv'), index=False)

    edge_df = pd.DataFrame(positive_edges, columns=['Gene1', 'Gene2'])
    edge_df['Gene1'] = edge_df['Gene1'].apply(lambda x: f'Gene_{x}')
    edge_df['Gene2'] = edge_df['Gene2'].apply(lambda x: f'Gene_{x}')
    edge_df.to_csv(os.path.join(split_dir, 'refNetwork.csv'), index=False)
    
    np.save(os.path.join(split_dir, 'expression.npy'), expression_data)
    np.save(os.path.join(split_dir, 'network.npy'), network_data)

def save_split_info(dataset_id, train_ratio, split_info, base_dir='./data/splits'):
    dataset_dir = os.path.join(base_dir, f'DS{dataset_id}')
    os.makedirs(dataset_dir, exist_ok=True)
    
    info_file = os.path.join(dataset_dir, f'split_info_{train_ratio:.2f}.json')
    with open(info_file, 'w') as f:
        json.dump(split_info, f, indent=4)

def plot_performance_curves(true_labels, pred_probs, save_dir, prefix=''):
    """
    Generate and save ROC curve, PR curve, and probability distribution plots
    
    Args:
        true_labels: numpy array of true binary labels
        pred_probs: numpy array of predicted probabilities
        save_dir: directory to save the plots
        prefix: prefix for the saved files (e.g., 'train', 'valid', 'test')
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{prefix} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{prefix}_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
    pr_auc = auc(recall, precision)
    
    # Calculate random baseline (proportion of positive samples)
    baseline = np.sum(true_labels) / len(true_labels)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.axhline(y=baseline, color='navy', linestyle='--', label=f'Baseline ({baseline:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{prefix} Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{prefix}_pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Probability Distribution
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=pred_probs[true_labels == 0], label='Negative Class', fill=True)
    sns.kdeplot(data=pred_probs[true_labels == 1], label='Positive Class', fill=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title(f'{prefix} Prediction Probability Distribution')
    plt.legend()
    plt.grid(True)
    
    # Add text box with class proportions
    total = len(true_labels)
    pos_prop = np.sum(true_labels == 1) / total
    neg_prop = 1 - pos_prop
    
    plt.text(0.02, 0.98, 
             f'Class Distribution:\nNegative: {neg_prop:.1%}\nPositive: {pos_prop:.1%}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(save_dir, f'{prefix}_prob_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Save summary statistics
    stats = {
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'Positive_Ratio': pos_prop,
        'Mean_Positive_Prob': np.mean(pred_probs[true_labels == 1]),
        'Mean_Negative_Prob': np.mean(pred_probs[true_labels == 0]),
        'Median_Positive_Prob': np.median(pred_probs[true_labels == 1]),
        'Median_Negative_Prob': np.median(pred_probs[true_labels == 0])
    }
    
    with open(os.path.join(save_dir, f'{prefix}_stats.txt'), 'w') as f:
        for key, value in stats.items():
            f.write(f'{key}: {value:.4f}\n')

def visualize_all_splits(train_labels, train_probs, valid_labels, valid_probs, 
                        test_labels, test_probs, save_dir):
    for labels, probs, prefix in [
        (train_labels, train_probs, 'train'),
        (valid_labels, valid_probs, 'valid'),
        (test_labels, test_probs, 'test')
    ]:
        plot_performance_curves(labels, probs, save_dir, prefix)

# %%
def run_experiment(gpu_id, dataset_id, train_ratio, embeddings,
                   input_dim, output_dim, num_epochs, batch_size, learning_rate,
                   loss_function, embedding_type, negative_ratio, temperature,
                   result_queue, n_runs=1):
    if torch.cuda.is_available() and gpu_id is not None:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device('cpu')
    
    datasets = get_datasets()
    dataset_info = next((dataset for dataset in datasets if dataset['dataset_id'] == dataset_id), None)
    print(f"Dataset Info: {dataset_info}")
    if dataset_info is None:
        raise ValueError(f"Dataset ID {dataset_id} not found in available datasets")
    dataset_id = dataset_info['dataset_id']
    print(f"\nProcessing Dataset {dataset_id}...")
    ds_clean, ds_noisy, gene_names = load_data(dataset_info)

    num_genes = ds_noisy.shape[0]
    num_cells = ds_noisy.shape[1]

    if dataset_id < 1000:
        # simulated data
        gt_grn_file = f'../SERGIO/data_sets/{dataset_info["folder_name"]}/gt_GRN.csv'
        H = load_ground_truth_grn(gt_grn_file)
    else:
        # real data
        # print(f"gene_names: {gene_names}")
        gt_grn_file = dataset_info['network_file']
        H = load_ground_truth_grn(gt_grn_file, gene_names=gene_names)
    
    tf_indices = identify_tfs(H)
    print(f"\nIdentified TFs: {len(tf_indices)} out of {H.shape[0]} genes")
    print(f"TF indices: {tf_indices}")
    
    valid_ratio = (1-train_ratio)/2
    test_ratio = (1-train_ratio)/2
    
    G = sample_partial_grn(H, sample_ratio=train_ratio + valid_ratio)
    G_train, G_valid = split_train_valid(G, train_ratio=train_ratio/(train_ratio+valid_ratio))
    G_test = get_test_set(H, G)
    # G_train_valid = G_train + G_valid
    
    print("\nSampled Matrix H:")
    print(f"Shape: {H.shape}")
    print(f"Number of 1s: {np.sum(H == 1)}")
    print(f"Number of 0s: {np.sum(H == 0)}")
    print(f"Number of -1s: {np.sum(H == -1)}")
        
    print("\nSampled Matrix G:")
    print(f"Shape: {G.shape}")
    print(f"Number of 1s: {np.sum(G == 1)}")
    print(f"Number of 0s: {np.sum(G == 0)}")
    print(f"Number of -1s: {np.sum(G == -1)}")

    print("\nTraining Matrix G_train:")
    print(f"Shape: {G_train.shape}")
    print(f"Number of 1s: {np.sum(G_train == 1)}")
    print(f"Number of 0s: {np.sum(G_train == 0)}")
    print(f"Number of -1s: {np.sum(G_train == -1)}")

    print("\nValidation Matrix G_valid:")
    print(f"Shape: {G_valid.shape}")
    print(f"Number of 1s: {np.sum(G_valid == 1)}")
    print(f"Number of 0s: {np.sum(G_valid == 0)}")
    print(f"Number of -1s: {np.sum(G_valid == -1)}")

    print("\nTest Matrix G_test:")
    print(f"Shape: {G_test.shape}")
    print(f"Number of 1s: {np.sum(G_test == 1)}")
    print(f"Number of 0s: {np.sum(G_test == 0)}")
    print(f"Number of -1s: {np.sum(G_test == -1)}")
    
    split_info = {
        'train_ratio': train_ratio,
        'valid_ratio': valid_ratio,
        'test_ratio': test_ratio,
        'num_genes': num_genes,
        'num_cells': num_cells,
        'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    save_split_info(dataset_id, train_ratio, split_info)

    for split_name, exp_data, net_data in [
        ('train', ds_noisy, G_train),
        ('valid', ds_noisy, G_valid),
        ('test', ds_noisy, G_test)
    ]:
        save_split_data(dataset_id, split_name, exp_data, net_data)
        split_dir = os.path.join('./data/splits', f'DS{dataset_id}', split_name)
        normalized_data = pd.DataFrame(exp_data, 
                                     index=[f'Gene_{i}' for i in range(exp_data.shape[0])],
                                     columns=[f'Cell_{i}' for i in range(exp_data.shape[1])])
        normalized_data.to_csv(os.path.join(split_dir, 'bin-normalized-matrix.csv'))

    
    all_train_metrics = []
    all_valid_metrics = []
    all_test_metrics = []
    best_model = None
    best_valid_auc = -1

    for run in range(n_runs):
        # Set different seed for each run
        run_seed = 42 + run
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        seed_everything(run_seed)
    
        if loss_function == 'SNN':
            model = train_snn(embeddings, G_train, input_dim, output_dim, num_epochs, batch_size, learning_rate, negative_ratio, temperature, device)
        elif loss_function == 'CL':
            model = train_cl(embeddings, G_train, input_dim, output_dim, num_epochs, batch_size, learning_rate, negative_ratio, device)
        elif loss_function == 'CEL':
            model = train_cel(embeddings, G_train, input_dim, output_dim, num_epochs, batch_size, learning_rate, negative_ratio, device)
        elif loss_function == 'BCE':
            model = train_bce(embeddings, G_train, input_dim, output_dim, num_epochs, batch_size, learning_rate, negative_ratio, device)
        
        train_metrics = evaluate_contrastive_model(model, embeddings, G_train, loss_function, device)
        # print("Train Metrics: ", train_metrics)
        # plot precision and recall curve
        plt.figure(figsize=(12, 6))
        plt.plot(train_metrics['recall_curve'], train_metrics['precision_curve'], label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Train Precision-Recall curve')
        plt.legend()
        plt.savefig(f'./results/cl/DS{dataset_id}/train_precision_recall_curve.png')
        valid_metrics = evaluate_contrastive_model(model, embeddings, G_valid, loss_function, device)
        # print("Validation Metrics: ", valid_metrics)
        plt.figure(figsize=(12, 6))
        plt.plot(valid_metrics['recall_curve'], valid_metrics['precision_curve'], label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Validation Precision-Recall curve')
        plt.legend()
        plt.savefig(f'./results/cl/DS{dataset_id}/valid_precision_recall_curve.png')
        test_metrics = evaluate_contrastive_model(model, embeddings, G_test, loss_function, device)
        # print("Test Metrics: ", test_metrics)
        plt.figure(figsize=(12, 6))
        plt.plot(test_metrics['recall_curve'], test_metrics['precision_curve'], label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Test Precision-Recall curve')
        plt.legend()
        plt.savefig(f'./results/cl/DS{dataset_id}/test_precision_recall_curve.png')

        all_train_metrics.append(train_metrics)
        all_valid_metrics.append(valid_metrics)
        all_test_metrics.append(test_metrics)

        if valid_metrics['auc_roc'] > best_valid_auc:
            best_valid_auc = valid_metrics['auc_roc']
            best_model = model

        metric_keys = ['auc_roc', 'auc_pr', 'f1_score', 'mcc', 'k_precision', 'k_recall']
        final_metrics = {
            'train': {'means': {}, 'stds': {}},
            'valid': {'means': {}, 'stds': {}},
            'test': {'means': {}, 'stds': {}}
        }

        for metric in metric_keys:
            # Training metrics
            values = [m[metric] for m in all_train_metrics]
            final_metrics['train']['means'][metric] = np.mean(values)
            final_metrics['train']['stds'][metric] = np.std(values)
            
            # Validation metrics
            values = [m[metric] for m in all_valid_metrics]
            final_metrics['valid']['means'][metric] = np.mean(values)
            final_metrics['valid']['stds'][metric] = np.std(values)
            
            # Test metrics
            values = [m[metric] for m in all_test_metrics]
            final_metrics['test']['means'][metric] = np.mean(values)
            final_metrics['test']['stds'][metric] = np.std(values)
        
        # Use the best model for final embeddings
        best_model.eval()
        with torch.no_grad():
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
            projected_embeddings = best_model.projection(embeddings_tensor).cpu().numpy()

    save_dir = f'./results/visualizations/DS{dataset_id}/{embedding_type}_{loss_function}'

    visualize_all_splits(
        train_metrics['true_labels'], train_metrics['probabilities'],
        valid_metrics['true_labels'], valid_metrics['probabilities'],
        test_metrics['true_labels'], test_metrics['probabilities'],
        save_dir
    )

    result_queue.put({
        'dataset_id': dataset_id,
        'train_ratio': train_ratio,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'loss_function': loss_function,
        'embedding_type': embedding_type,
        'negative_ratio': negative_ratio,
        'temperature': temperature if loss_function == 'SNN' else None,
        'train_auc': f"{final_metrics['train']['means']['auc_roc']:.4f} +- {final_metrics['train']['stds']['auc_roc']:.4f}",
        'valid_auc': f"{final_metrics['valid']['means']['auc_roc']:.4f} +- {final_metrics['valid']['stds']['auc_roc']:.4f}",
        'test_auc': f"{final_metrics['test']['means']['auc_roc']:.4f} +- {final_metrics['test']['stds']['auc_roc']:.4f}",
        'train_pr_auc': f"{final_metrics['train']['means']['auc_pr']:.4f} +- {final_metrics['train']['stds']['auc_pr']:.4f}",
        'valid_pr_auc': f"{final_metrics['valid']['means']['auc_pr']:.4f} +- {final_metrics['valid']['stds']['auc_pr']:.4f}",
        'test_pr_auc': f"{final_metrics['test']['means']['auc_pr']:.4f} +- {final_metrics['test']['stds']['auc_pr']:.4f}",
        'train_f1': f"{final_metrics['train']['means']['f1_score']:.4f} +- {final_metrics['train']['stds']['f1_score']:.4f}",
        'valid_f1': f"{final_metrics['valid']['means']['f1_score']:.4f} +- {final_metrics['valid']['stds']['f1_score']:.4f}",
        'test_f1': f"{final_metrics['test']['means']['f1_score']:.4f} +- {final_metrics['test']['stds']['f1_score']:.4f}",
        'train_k_precision': f"{final_metrics['train']['means']['k_precision']:.4f} +- {final_metrics['train']['stds']['k_precision']:.4f}",
        'valid_k_precision': f"{final_metrics['valid']['means']['k_precision']:.4f} +- {final_metrics['valid']['stds']['k_precision']:.4f}",
        'test_k_precision': f"{final_metrics['test']['means']['k_precision']:.4f} +- {final_metrics['test']['stds']['k_precision']:.4f}",
        'train_mcc': f"{final_metrics['train']['means']['mcc']:.4f} +- {final_metrics['train']['stds']['mcc']:.4f}",
        'valid_mcc': f"{final_metrics['valid']['means']['mcc']:.4f} +- {final_metrics['valid']['stds']['mcc']:.4f}",
        'test_mcc': f"{final_metrics['test']['means']['mcc']:.4f} +- {final_metrics['test']['stds']['mcc']:.4f}",
        'train_k_recall': f"{final_metrics['train']['means']['k_recall']:.4f} +- {final_metrics['train']['stds']['k_recall']:.4f}",
        'valid_k_recall': f"{final_metrics['valid']['means']['k_recall']:.4f} +- {final_metrics['valid']['stds']['k_recall']:.4f}",
        'test_k_recall': f"{final_metrics['test']['means']['k_recall']:.4f} +- {final_metrics['test']['stds']['k_recall']:.4f}",
        # 'train_confusion_matrix': train_metrics['confusion_matrix'],
        # 'valid_confusion_matrix': valid_metrics['confusion_matrix'],
        # 'test_confusion_matrix': test_metrics['confusion_matrix'],
        # 'train_precision_curve': train_metrics['precision_curve'],
        # 'train_recall_curve': train_metrics['recall_curve'],
        # 'train_thresholds': train_metrics['thresholds'],
        # 'valid_precision_curve': valid_metrics['precision_curve'],
        # 'valid_recall_curve': valid_metrics['recall_curve'],
        # 'valid_thresholds': valid_metrics['thresholds'],
        # 'test_precision_curve': test_metrics['precision_curve'],
        # 'test_recall_curve': test_metrics['recall_curve'],
        # 'test_thresholds': test_metrics['thresholds'],
        'train_labels': train_metrics['true_labels'],
        'train_scores': train_metrics['probabilities'],
        'valid_labels': valid_metrics['true_labels'],
        'valid_scores': valid_metrics['probabilities'],
        'test_labels': test_metrics['true_labels'],
        'test_scores': test_metrics['probabilities'],
        'model_state_dict': best_model.state_dict(),
        'projected_embeddings': projected_embeddings
    })

def logger_process(result_queue, num_experiments):
    experiments_logged = 0
    while experiments_logged < num_experiments:
        try:
            result = result_queue.get(timeout=1)
            log_dir = log_experiment(result)
            print(f"Experiment logged in {log_dir}")
            experiments_logged += 1
        except queue.Empty:
            continue

import random

def process_batch(batch):
    processes = []
    for args in batch:
        gpu_id, embedding_type, output_dim, num_epochs, train_ratio, dataset_id, batch_size, learning_rate, loss_function, negative_ratio, temperature = args
        embeddings_path = f'./results/cl/DS{dataset_id}/{embedding_type}_embeddings.npy'
        embeddings = np.load(embeddings_path)
        input_dim = embeddings.shape[1]
        p = multiprocessing.Process(target=run_experiment, args=(
            gpu_id, dataset_id, train_ratio, embeddings,
            input_dim, output_dim, num_epochs, batch_size, learning_rate,
            loss_function, embedding_type, negative_ratio, temperature, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    # Set random seed even though this does not guarantee complete reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()

    # single parameters
    embedding_types = ['pca']  # ['pca', 'vae']
    output_dims = [16] # [16, 32, 64, 128, 256]
    num_epochs_list = [50] # [5, 10, 50, 100, 200]
    train_ratios = [0.8]
    dataset_ids = [2] # 1,2,3,1001,1002,1003,1004,1005,1006
    batch_sizes = [32] # [16, 32, 64]
    learning_rates = [1e-3] # [1e-3, 5e-3, 1e-4, 5e-4]
    loss_functions = ['SNN']  # ['BCE', 'CEL', 'CL', 'SNN']
    negative_ratios = [50] # [5, 10, 20, 50, 100, 200, 500]
    temperatures = [1]

    # embedding_types = ['pca']  # ['pca', 'vae']
    # output_dims = [16, 32, 64] # [16, 32, 64, 128, 256]
    # num_epochs_list = [10, 50, 200] # [5, 10, 50, 100, 200]
    # batch_sizes = [32, 64] # [16, 32, 64]z
    # learning_rates = [1e-3, 1e-4, 5e-4] # [1e-3, 5e-3, 1e-4, 5e-4]
    # loss_functions = ['SNN']  # ['BCE', 'CEL', 'CL', 'SNN']
    # negative_ratios = [10, 50, 100, 500] # [5, 10, 20, 50, 100, 200, 500]
    # temperatures = [0.1, 1.0, 10, 100]  # [0.1, 0.5, 1.0, 2.0, 5.0]

    experiments = []
    for embedding_type, output_dim, num_epochs, train_ratio, dataset_id, batch_size, learning_rate, loss_function, negative_ratio in product(
        embedding_types, output_dims, num_epochs_list, train_ratios, dataset_ids, batch_sizes, learning_rates, loss_functions, negative_ratios
    ):
        if loss_function == 'SNN':
            for temperature in temperatures:
                experiments.append((embedding_type, output_dim, num_epochs, train_ratio, dataset_id, batch_size, learning_rate, loss_function, negative_ratio, temperature))
        else:
            experiments.append((embedding_type, output_dim, num_epochs, train_ratio, dataset_id, batch_size, learning_rate, loss_function, negative_ratio, None))

    print(f"Running {len(experiments)} experiments on {num_gpus} GPUs.")
    print(experiments)

    experiments_with_gpus = []
    for idx, exp in enumerate(experiments):
        if num_gpus > 0:
            gpu_id = idx % num_gpus
        else:
            gpu_id = None
        experiments_with_gpus.append((gpu_id,) + exp)
    
    result_queue = multiprocessing.Queue()
    logger = multiprocessing.Process(target=logger_process, args=(result_queue, len(experiments)))
    logger.start()
    
    batch_process_size = max(num_gpus * 2, 1)
    for i in range(0, len(experiments_with_gpus), batch_process_size):
        batch = experiments_with_gpus[i:i+batch_process_size]
        process_batch(batch)

    logger.join()
    print("All experiments completed and logged.")
