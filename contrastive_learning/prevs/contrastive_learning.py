# %%
import math
import sys
import os
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, silhouette_score
import os
import sys
from tqdm import tqdm
current_dir = os.getcwd()
from GENIE3.GENIE3 import *
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.manifold import TSNE
import networkx as nx

sys.path.append('./baselines')
sys.path.append('../SERGIO')
nthreads=12

# %%
def get_clusters_from_adj(adj):
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    clusters = list(nx.weakly_connected_components(G))
    cluster_labels = {}
    for i, cluster in enumerate(clusters):
        for gene in cluster:
            cluster_labels[gene] = i
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
    return sorted(datasets, key=lambda x: x['dataset_id'])

def load_data(dataset_info):
    dataset_id = dataset_info['dataset_id']
    data_dir = f'../SERGIO/imputation_data_2/DS{dataset_id}/'
    ds_clean_path = os.path.join(data_dir, 'DS6_clean.npy')
    ds_noisy_path = os.path.join(data_dir, 'DS6_45_iter_0.npy')
    if not os.path.exists(ds_clean_path) or not os.path.exists(ds_noisy_path):
        print(f"Data files not found for Dataset {dataset_id}. Skipping.")
        return None, None
    ds_clean = np.load(ds_clean_path).astype(np.float32)
    ds_noisy = np.load(ds_noisy_path).astype(np.float32)
    return ds_clean, ds_noisy

def get_indirected_adjacency(file_path, num_genes):
    direct_connections = {i: set() for i in range(num_genes)}
    with open(file_path, 'r') as f:
        for line in f:
            source, target = map(int, line.strip().split(','))
            direct_connections[source].add(target)
    adj_matrix = np.zeros((num_genes, num_genes), dtype=float)
    for source in range(num_genes):
        for target in direct_connections[source]:
            adj_matrix[source, target] = 1.0
        for intermediate in direct_connections[source]:
            for indirect_target in direct_connections[intermediate]:
                if indirect_target != source and indirect_target not in direct_connections[source]:
                    adj_matrix[source, indirect_target] = 0.5
    return adj_matrix

def load_ground_truth_grn(file_path, num_genes):
    H = np.zeros((num_genes, num_genes))
    with open(file_path, 'r') as f:
        for line in f:
            source, target = map(int, line.strip().split(','))
            H[source, target] = 1
    return H

def sample_partial_grn(H, sample_ratio=0.9):
    num_edges = np.sum(H)
    num_sample = int(num_edges * sample_ratio)
    edge_indices = np.argwhere(H == 1)
    np.random.shuffle(edge_indices)
    sampled_edges = edge_indices[:num_sample]
    G = np.zeros_like(H)
    G[tuple(zip(*sampled_edges))] = 1
    return G

def split_train_valid(G, train_ratio=0.8889):  # 8:1 ratio
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

def compute_validation_loss(model, ds_tensor, w_ij_valid, temperature):
    model.eval()
    with torch.no_grad():
        recon_data, _, _, z = model(ds_tensor)
        # Compute distances
        dot_product = torch.matmul(z, z.T)
        square_sum = torch.sum(z ** 2, dim=1, keepdim=True)
        distances = square_sum + square_sum.T - 2 * dot_product
        distances = torch.clamp(distances, min=0.0)
        # Compute contrastive loss using w_ij_valid
        numerator = torch.exp(-distances / temperature) * w_ij_valid
        denominator = torch.exp(-distances / temperature)
        loss_matrix = -torch.log((torch.sum(numerator, dim=1) / torch.sum(denominator, dim=1)) + 1e-8)
        contrastive_loss = torch.mean(loss_matrix)
    return contrastive_loss.item()

# def compute_validation_loss(model, ds_tensor, w_ij_valid, temperature):
#     model.eval()
#     with torch.no_grad():
#         recon_data, _, _, z = model(ds_tensor)
#         z_norm = F.normalize(z, p=2, dim=1)
#         similarities = torch.matmul(z_norm, z_norm.T)
#         positive_mask = (w_ij_valid > 0).float()
#         negative_mask = 1 - positive_mask
#         pos_loss = -torch.log(torch.exp(similarities / temperature) + 1e-8)
#         pos_loss = (pos_loss * positive_mask).sum(1) / positive_mask.sum(1).clamp(min=1)
#         neg_loss = -torch.log(1 - torch.exp(similarities / temperature) + 1e-8)
#         neg_loss = (neg_loss * negative_mask).sum(1) / negative_mask.sum(1).clamp(min=1)
#         contrastive_loss = (pos_loss + neg_loss).mean()
#         return contrastive_loss.item()

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

# %%
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim):
        super(VAE, self).__init__()

        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh())
        # Latent variables
        self.mu = nn.Linear(hidden_size, latent_dim)
        self.logvar = nn.Linear(hidden_size, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_dim),
            nn.Tanh())

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparametrize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar, z

def substitute_dataset(ds):
    ds[ds == 0] = np.nan
    for i in tqdm(range(9)):
        ds_cell_type = ds[:,i*300:(i+1)*300]
        mean_array = np.nanmean(ds_cell_type, axis=1)
        var_array = np.nanvar(ds_cell_type, axis=1)
        for j in range(100):
            ds_cell_type[j,:] = np.random.normal(loc=mean_array[j], scale=np.sqrt(var_array[j]), size=300)
    ds[ds<0] = 0.0
    np.nan_to_num(ds, copy=False)
    return(ds)

def contrastive_imputation(ds, G_train, G_valid, dataset_id, embedding_dim=128, num_epochs=3000, batch_size=256, 
                           learning_rate=1e-4, lambda_recon=1.0, lambda_kld=0.01, lambda_edge=0.5,
                           validation_interval=1, temperature=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = ds.copy()
    num_genes, num_cells = ds.shape

    # Basic imputation
    print("1", ds)
    ds = substitute_dataset(ds)
    print("2", ds)
    # Normalize data
    ds_min = np.min(ds)
    ds_max = np.max(ds)
    ds_norm = (ds - ds_min) / (ds_max - ds_min)
    # Scale to [-1, 1] for Tanh activation
    ds_norm = ds_norm * 2 - 1
    print("3", ds_norm)
    ds_tensor = torch.tensor(ds_norm, dtype=torch.float)
    ds_tensor = ds_tensor.to(device)
    input_dim = ds_tensor.shape[1]
    model = VAE(input_dim, hidden_size=1024, latent_dim=embedding_dim).to(device)
    w_ij_train = torch.tensor(G_train, dtype=torch.float).to(device)
    w_ij_valid = torch.tensor(G_valid, dtype=torch.float).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    num_samples = ds_tensor.size(0)
    num_batches = math.ceil(num_samples / batch_size)
    os.makedirs(f'./results/cl/DS{dataset_id}/', exist_ok=True)
    log_file = open(f'./results/cl/DS{dataset_id}/training_logs.txt', 'w')
    
    param_prev = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}

    training_losses = []
    validation_losses = []

    # Precompute positive and negative edges
    positive_edges = np.argwhere(G_train == 1)
    negative_edges = np.argwhere(G_train == 0)

    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            batch_indices = permutation[i * batch_size : (i + 1) * batch_size]
            batch_data = ds_tensor[batch_indices, :]  # Shape: [batch_size, num_cells]
            recon_batch, mu, logvar, z = model(batch_data)  # z shape: [batch_size, latent_dim]
            batch_gene_indices = batch_indices.cpu().numpy()
            mask_positive = np.isin(positive_edges[:, 0], batch_gene_indices) & np.isin(positive_edges[:, 1], batch_gene_indices)
            positive_edges_in_batch = positive_edges[mask_positive]
            mask_negative = np.isin(negative_edges[:, 0], batch_gene_indices) & np.isin(negative_edges[:, 1], batch_gene_indices)
            negative_edges_in_batch = negative_edges[mask_negative]
            num_positive = len(positive_edges_in_batch)
            if num_positive == 0:
                continue
            num_negative = num_positive
            if len(negative_edges_in_batch) < num_negative:
                num_negative = len(negative_edges_in_batch)
            if num_negative == 0:
                continue
            sampled_negative_edges = negative_edges_in_batch[np.random.choice(len(negative_edges_in_batch), size=num_negative, replace=False)]
            edge_indices_in_batch = np.vstack((positive_edges_in_batch, sampled_negative_edges))
            edge_labels = np.hstack((np.ones(num_positive), np.zeros(num_negative)))

            # Map global gene indices to batch-relative indices
            gene_to_batch_idx = {gene: idx for idx, gene in enumerate(batch_gene_indices)}
            try:
                batch_edge_indices = np.array([[gene_to_batch_idx[edge[0]], gene_to_batch_idx[edge[1]]] for edge in edge_indices_in_batch])
            except KeyError as e:
                print(f"Gene index not in batch: {e}")
                continue

            # Convert to tensors
            edge_indices_tensor = torch.tensor(batch_edge_indices, dtype=torch.long).to(device)  # Shape: [2*num_positive, 2]
            edge_labels_tensor = torch.tensor(edge_labels, dtype=torch.float).to(device)  # Shape: [2*num_positive]

            # Extract embeddings for the edge pairs
            embedding_i = z[edge_indices_tensor[:, 0]]  # Shape: [2*num_positive, latent_dim]
            embedding_j = z[edge_indices_tensor[:, 1]]  # Shape: [2*num_positive, latent_dim]

            # Compute edge scores
            edge_scores = edge_score(embedding_i, embedding_j)  # Shape: [2*num_positive]
            # BCE loss on edge scores
            edge_loss = F.binary_cross_entropy(edge_scores, edge_labels_tensor)

            # Reconstruction and KLD loss
            recon_loss = F.mse_loss(recon_batch, batch_data)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Contrastive Loss
            dot_product = torch.matmul(z, z.T)  # Shape: [batch_size, batch_size]
            square_sum = torch.sum(z ** 2, dim=1, keepdim=True)  # Shape: [batch_size, 1]
            distances = square_sum + square_sum.T - 2 * dot_product  # Shape: [batch_size, batch_size]
            distances = torch.clamp(distances, min=0.0)

            batch_w_ij = w_ij_train[batch_indices][:, batch_indices]  # Shape: [batch_size, batch_size]
            numerator = torch.exp(-distances / temperature) * batch_w_ij  # Shape: [batch_size, batch_size]
            denominator = torch.exp(-distances / temperature)  # Shape: [batch_size, batch_size]
            loss_matrix = -torch.log((torch.sum(numerator, dim=1) / torch.sum(denominator, dim=1)) + 1e-8)
            contrastive_loss = torch.mean(loss_matrix)
            # print("batch_w_ij: ", batch_w_ij)

            # loss = lambda_recon * recon_loss + lambda_kld * kld_loss + contrastive_loss
            loss = recon_loss + lambda_kld * kld_loss

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Logging
        avg_loss = epoch_loss / num_batches
        training_losses.append(avg_loss)
        log_file.write(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}\n')
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}')

        # Validation
        if (epoch + 1) % validation_interval == 0:
            val_loss = compute_validation_loss(model, ds_tensor, w_ij_valid, temperature)
            validation_losses.append(val_loss)
            log_file.write(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}\n')
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
            scheduler.step(val_loss)
        
        # Compute Parameter Change
        param_change = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param_change += torch.sum(torch.abs(param.data - param_prev[name]))
        param_prev = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
        print(f"Epoch [{epoch+1}/{num_epochs}], Total Parameter Change: {param_change.item()}")
        log_file.write(f"Epoch [{epoch+1}/{num_epochs}], Total Parameter Change: {param_change.item()}\n")
    
    # Save Training Logs and Model
    np.save(f'./results/cl/DS{dataset_id}/training_losses.npy', np.array(training_losses))
    np.save(f'./results/cl/DS{dataset_id}/validation_losses.npy', np.array(validation_losses))

    model.eval()
    with torch.no_grad():
        recon_data, _, _, z = model(ds_tensor)
        ds_imputed = recon_data.cpu().numpy()
        embeddings = z.cpu().numpy()

    # Rescale Imputed Data
    ds_imputed = (ds_imputed + 1) / 2 * (ds_max - ds_min) + ds_min
    ds_imputed[ds_imputed < 0] = 0.0

    log_file.close()
    model_save_path = f'./models/contrastive_vae_model_DS{dataset_id}.pth'
    os.makedirs('./models/', exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    return ds_imputed, embeddings

# %%
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
    plt.show()

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
    plt.show()

def run_pipeline(ds_noisy, ds_clean, interactions, G_train, G_valid, G_test, G, H, dataset_id):
    ds_imputed, embeddings = contrastive_imputation(ds_noisy, G_train, G_valid, dataset_id)

    mse = mean_squared_error(ds_clean.flatten(), ds_imputed.flatten())
    print(f"MSE between Clean and Imputed Data: {mse:.4f}")
    log_file.write(f"MSE between Clean and Imputed Data: {mse:.4f}\n")

    ds_imputed_T = ds_imputed.T
    
    VIM_imputed = GENIE3(ds_imputed_T, nthreads=12, ntrees=100, regulators='all', gene_names=[str(s) for s in range(ds_imputed_T.shape[1])])
    
    inferred_grn = VIM_imputed.flatten()
    
    roc_auc_train = roc_auc_score(G_train.flatten(), inferred_grn)
    roc_auc_valid = roc_auc_score(G_valid.flatten(), inferred_grn)
    roc_auc_test = roc_auc_score(G_test.flatten(), inferred_grn)
    roc_auc_total = roc_auc_score(G.flatten(), inferred_grn)
    roc_auc_total_traditional = roc_auc_score(interactions.flatten(), inferred_grn)
    
    print(f"ROC AUC Score on Training Set: {roc_auc_train:.4f}")
    print(f"ROC AUC Score on Validation Set: {roc_auc_valid:.4f}")
    print(f"ROC AUC Score on Test Set: {roc_auc_test:.4f}")
    print(f"ROC AUC Score on Total G (Train + Valid): {roc_auc_total:.4f}\n")
    print(f"ROC AUC Score on Total G (Train + Valid) using traditional GENIE3: {roc_auc_total_traditional:.4f}\n")
    
    log_file.write(f"ROC AUC Score on Training Set: {roc_auc_train:.4f}\n")
    log_file.write(f"ROC AUC Score on Validation Set: {roc_auc_valid:.4f}\n")
    log_file.write(f"ROC AUC Score on Test Set: {roc_auc_test:.4f}\n")
    log_file.write(f"ROC AUC Score on Total G (Train + Valid): {roc_auc_total:.4f}\n")
    log_file.write(f"ROC AUC Score on Total G (Train + Valid) using traditional GENIE3: {roc_auc_total_traditional:.4f}\n")
    
    roc_auc_full = roc_auc_score(H.flatten(), inferred_grn)
    print(f"ROC AUC Score on Full Ground Truth H: {roc_auc_full:.4f}\n")
    log_file.write(f"ROC AUC Score on Full Ground Truth H: {roc_auc_full:.4f}\n")

    results = {
        'roc_auc_train': roc_auc_train,
        'roc_auc_valid': roc_auc_valid,
        'roc_auc_test': roc_auc_test,
        'roc_auc_total': roc_auc_total,
        'roc_auc_full': roc_auc_full,
        'roc_auc_total_traditional': roc_auc_total_traditional,
        'mse': mse,
        'embeddings': embeddings
    }
    return results

datasets = get_datasets()
# for dataset_info in datasets:
dataset_info = datasets[1]
# Main processing loop
dataset_id = dataset_info['dataset_id']
print(f"\nProcessing Dataset {dataset_id}...")
ds_clean, ds_noisy = load_data(dataset_info)

num_genes = ds_noisy.shape[0]
cells_per_type = dataset_info['cells_per_type']
num_cells = ds_noisy.shape[1]

gt_grn_file = f'../SERGIO/data_sets/{dataset_info["folder_name"]}/gt_GRN.csv'
H = load_ground_truth_grn(gt_grn_file, num_genes)
G = sample_partial_grn(H, sample_ratio=0.9)
G_train, G_valid = split_train_valid(G, train_ratio=8/9)  # 8:1 ratio
G_test = get_test_set(H, G)

target_file = f'../SERGIO/data_sets/{dataset_info["folder_name"]}/Interaction_cID_{dataset_info["dynamics"]}.txt'
interactions = load_interactions_info(num_genes, target_file)

log_dir = f'./results/cl/DS{dataset_id}'
os.makedirs(log_dir, exist_ok=True)
log_file_path = f'./results/cl/DS{dataset_id}/log.txt'

cluster_labels_train, cluster_labels_valid, cluster_labels_test, cluster_labels_H = plot_grn_from_graphs(G_train, G_valid, G_test, H, dataset_id)

with open(log_file_path, 'w') as log_file:
    # Evaluate clean data
    print("Evaluating Clean Data...")
    ds_clean_T = ds_clean.T
    VIM_clean = GENIE3(ds_clean_T, nthreads=12, ntrees=100, regulators='all',
                    gene_names=[str(s) for s in range(ds_clean_T.shape[1])])
    roc_auc_clean = roc_auc_score(interactions.flatten(), VIM_clean.flatten())
    print(f"ROC AUC Score for Clean Data: {roc_auc_clean:.4f}\n")
    log_file.write(f"ROC AUC Score for Clean Data: {roc_auc_clean:.4f}\n")

    # Evaluate noisy dataƒ
    print("Evaluating Noisy Data...")
    ds_noisy_T = ds_noisy.T
    VIM_noisy = GENIE3(ds_noisy_T, nthreads=12, ntrees=100, regulators='all',
                    gene_names=[str(s) for s in range(ds_noisy_T.shape[1])])
    roc_auc_noisy = roc_auc_score(interactions.flatten(), VIM_noisy.flatten())
    print(f"ROC AUC Score for Noisy Data: {roc_auc_noisy:.4f}\n")
    log_file.write(f"ROC AUC Score for Noisy Data: {roc_auc_noisy:.4f}\n")

    # Compute MSE between noisy data and clean data
    mse_noisy = mean_squared_error(ds_clean.flatten(), ds_noisy.flatten())
    print(f"MSE between Noisy Data and Clean Data: {mse_noisy:.4f}\n")
    log_file.write(f"MSE between Noisy Data and Clean Data: {mse_noisy:.4f}\n")

    print("Running Imputation and Analysis Pipeline...")
    results = run_pipeline(ds_noisy, ds_clean, interactions, G_train, G_valid, G_test, G, H, dataset_id)
    embeddings = results['embeddings']
    
    edge_probabilities = compute_edge_likelihoods(embeddings)
    valid_edge_indices = np.argwhere(G_valid == 1)
    test_edge_indices = np.argwhere(G_test == 1)
    valid_connected_probs = edge_probabilities[valid_edge_indices[:, 0], valid_edge_indices[:, 1]]
    valid_non_connected_probs = edge_probabilities[G_valid == 0]
    test_connected_probs = edge_probabilities[test_edge_indices[:, 0], test_edge_indices[:, 1]]
    test_non_connected_probs = edge_probabilities[G_test == 0]
    avg_valid_connected_prob = np.mean(valid_connected_probs)
    avg_valid_non_connected_prob = np.mean(valid_non_connected_probs)
    avg_test_connected_prob = np.mean(test_connected_probs)
    avg_test_non_connected_prob = np.mean(test_non_connected_probs)
    print(f"Validation Set - Connected Avg Prob: {avg_valid_connected_prob:.4f}, Non-Connected Avg Prob: {avg_valid_non_connected_prob:.4f}")
    print(f"Test Set - Connected Avg Prob: {avg_test_connected_prob:.4f}, Non-Connected Avg Prob: {avg_test_non_connected_prob:.4f}")
    t_stat_valid, p_value_valid = ttest_ind(valid_connected_probs, valid_non_connected_probs)
    print(f"Validation Set - T-test: t-statistic = {t_stat_valid:.4f}, p-value = {p_value_valid:.4e}")
    t_stat_test, p_value_test = ttest_ind(test_connected_probs, test_non_connected_probs)
    print(f"Test Set - T-test: t-statistic = {t_stat_test:.4f}, p-value = {p_value_test:.4e}")
    log_file.write(f"Validation Set - Connected Avg Prob: {avg_valid_connected_prob:.4f}, Non-Connected Avg Prob: {avg_valid_non_connected_prob:.4f}\n")
    log_file.write(f"Validation Set - T-test: t-statistic = {t_stat_valid:.4f}, p-value = {p_value_valid:.4e}\n")
    log_file.write(f"Test Set - Connected Avg Prob: {avg_test_connected_prob:.4f}, Non-Connected Avg Prob: {avg_test_non_connected_prob:.4f}\n")
    log_file.write(f"Test Set - T-test: t-statistic = {t_stat_test:.4f}, p-value = {p_value_test:.4e}\n")

    log_file.write(f"MSE between Clean and Imputed Data: {results['mse']:.4f}\n")
    print("Summary of ROC AUC Scores:")
    print(f"Clean Data: {roc_auc_clean:.4f}")
    print(f"Noisy Data: {roc_auc_noisy:.4f}")
    print(f"Imputed Data - Training Set: {results['roc_auc_train']:.4f}")
    print(f"Imputed Data - Validation Set: {results['roc_auc_valid']:.4f}")
    print(f"Imputed Data - Test Set: {results['roc_auc_test']:.4f}")
    print(f"Imputed Data - Total G (Train + Valid): {results['roc_auc_total']:.4f}")
    print(f"Imputed Data - Full Ground Truth H: {results['roc_auc_full']:.4f}")
    print(f"Total G (Train + Valid) using traditional GENIE3: {results['roc_auc_total_traditional']:.4f}")
    log_file.write("\nSummary of ROC AUC Scores:\n")
    log_file.write(f"Clean Data: {roc_auc_clean:.4f}\n")
    log_file.write(f"Noisy Data: {roc_auc_noisy:.4f}\n")
    log_file.write(f"Imputed Data - Training Set: {results['roc_auc_train']:.4f}\n")
    log_file.write(f"Imputed Data - Validation Set: {results['roc_auc_valid']:.4f}\n")
    log_file.write(f"Imputed Data - Test Set: {results['roc_auc_test']:.4f}\n")
    log_file.write(f"Imputed Data - Total G (Train + Valid): {results['roc_auc_total']:.4f}\n")
    log_file.write(f"Imputed Data - Full Ground Truth H: {results['roc_auc_full']:.4f}\n")
    log_file.write(f"Total G (Train + Valid) using traditional GENIE3: {results['roc_auc_total_traditional']:.4f}\n")

    print("\nSummary of MSE Values:")
    print(f"Noisy Data vs. Clean Data: {mse_noisy:.4f}")
    print(f"Imputed Data vs. Clean Data: {results['mse']:.4f}")
    log_file.write("\nSummary of MSE Values:\n")
    log_file.write(f"Noisy Data vs. Clean Data: {mse_noisy:.4f}\n")
    log_file.write(f"Imputed Data vs. Clean Data: {results['mse']:.4f}\n")

    print("Analyzing embeddings to see if connected nodes are closer...")
    pairwise_distances = squareform(pdist(embeddings, metric='euclidean'))
    connected_indices = np.argwhere(H == 1)
    non_connected_indices = np.argwhere(H == 0)

    # num_samples = 10000
    np.random.shuffle(connected_indices)
    np.random.shuffle(non_connected_indices)
    # connected_indices = connected_indices[:num_samples]
    # non_connected_indices = non_connected_indices[:num_samples]

    connected_distances = pairwise_distances[connected_indices[:, 0], connected_indices[:, 1]]
    non_connected_distances = pairwise_distances[non_connected_indices[:, 0], non_connected_indices[:, 1]]
    plt.figure(figsize=(10, 6))
    sns.kdeplot(connected_distances, label='Connected Nodes')
    sns.kdeplot(non_connected_distances, label='Non-Connected Nodes')
    plt.title('Distance Distribution between Node Embeddings')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'./results/cl/DS{dataset_id}/distance_distribution.png')
    plt.show()

    t_stat, p_value = ttest_ind(connected_distances, non_connected_distances)
    print(f"T-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}")
    log_file.write(f"T-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}\n")

    print("Performing t-SNE on embeddings...")
    plot_embeddings(embeddings, cluster_labels=cluster_labels_train, title='t-SNE Embeddings', save_path=f'./results/cl/DS{dataset_id}/tsne_embeddings.png')

    valid_indices = cluster_labels_H != -1
    valid_embeddings = embeddings[valid_indices]
    valid_labels = cluster_labels_H[valid_indices]
    unique_labels = np.unique(valid_labels)
    print(f"Unique Labels: {unique_labels}")
    print(f"Number of Unique Labels: {len(unique_labels)}")
    cluster_labels_H = get_clusters_from_adj(H)
    print(f"Cluster labels (cluster_labels_H): {cluster_labels_H}")
    cluster_labels_list_H = get_cluster_labels(cluster_labels_H, num_genes)
    num_clusters = len(set(cluster_labels_H.values()))
    print(f"Number of clusters: {num_clusters}")
    # sil_score = silhouette_score(valid_embeddings, valid_labels)
    # print(f"Silhouette Score: {sil_score:.4f}")
    # log_file.write(f"Silhouette Score: {sil_score:.4f}\n")