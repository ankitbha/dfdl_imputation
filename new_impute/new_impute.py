# %%
import sys
import os
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error
import os
import sys
from tqdm import tqdm
current_dir = os.getcwd()

# Import GENIE3
from GENIE3.GENIE3 import *

# Import imputation methods
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# For deep learning-based imputation (DeepImpute)
from baselines.deepimpute.deepimpute.multinet import MultiNet

# For graph convolutional networks
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import scprep
from scipy.sparse.linalg import bicgstab

# For SAUCIE
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# For MAGIC
from baselines.MAGIC.magic import magic

# For scScope
import baselines.scScope.scscope.scscope as scScope

nthreads = 12

# Ensure that the necessary paths are in sys.path
sys.path.append('./baselines')
sys.path.append('../SERGIO')

# %%
# Function to parse dataset information
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
    data_sets_dir = '../SERGIO/data_sets'  # Adjust the path if necessary
    for folder_name in os.listdir(data_sets_dir):
        dataset_info = parse_dataset_name(folder_name)
        if dataset_info:
            datasets.append(dataset_info)
    return sorted(datasets, key=lambda x: x['dataset_id'])

# Load your dataset
def load_data(dataset_info):
    dataset_id = dataset_info['dataset_id']
    num_genes = dataset_info['number_genes']
    num_bins = dataset_info['number_bins']
    cells_per_type = dataset_info['cells_per_type']
    num_cells = num_bins * cells_per_type

    data_dir = f'../SERGIO/imputation_data_2/DS{dataset_id}/'

    ds_clean_path = os.path.join(data_dir, 'DS6_clean.npy')
    ds_noisy_path = os.path.join(data_dir, 'DS6_45_iter_0.npy')

    if not os.path.exists(ds_clean_path) or not os.path.exists(ds_noisy_path):
        print(f"Data files not found for Dataset {dataset_id}. Skipping.")
        return None, None

    ds_clean = np.load(ds_clean_path).astype(np.float32)
    ds_noisy = np.load(ds_noisy_path).astype(np.float32)

    return ds_clean, ds_noisy

# Load ground truth network
def load_ground_truth(target_file, num_genes):
    gt = np.zeros((num_genes, num_genes))
    with open(target_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line_list = line.strip().split(',')
        target_index = int(float(line_list[0]))
        num_regs = int(float(line_list[1]))
        for i in range(num_regs):
            reg_index = int(float(line_list[i + 2]))
            gt[reg_index, target_index] = 1
    return gt

# Build the adjacency matrix
def build_adjacency_matrix(num_genes, interactions_file):
    adjacency_matrix = np.zeros((num_genes, num_genes))
    with open(interactions_file, 'r') as f:
        for line in f:
            tokens = line.strip().split(',')
            gene = int(float(tokens[0]))
            num_targets = int(float(tokens[1]))
            targets = [int(float(t)) for t in tokens[2:2 + num_targets]]
            for target in targets:
                adjacency_matrix[gene, target] = 1
    return adjacency_matrix

def knn_imputation(ds, n_neighbors=5, cells_per_type=300):
    # Replace zeros with NaN to mark missing values
    ds = ds.copy()
    ds[ds == 0] = np.nan
    
    # Initialize the imputed dataset
    ds_imputed = np.copy(ds)
    
    # Number of genes and cells
    num_genes, num_cells = ds.shape
    
    # Number of cell types
    num_cell_types = num_cells // cells_per_type
    
    # Loop over each cell type
    for i in range(num_cell_types):
        start_idx = i * cells_per_type
        end_idx = start_idx + cells_per_type
        ds_cell_type = ds[:, start_idx:end_idx]
        
        # Transpose the data to shape (cells, genes) for KNNImputer
        ds_cell_type_T = ds_cell_type.T
        
        # Initialize KNNImputer
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        
        # Perform imputation
        ds_cell_type_imputed_T = imputer.fit_transform(ds_cell_type_T)
        
        # Transpose back to original shape
        ds_cell_type_imputed = ds_cell_type_imputed_T.T
        
        # Update the imputed dataset
        ds_imputed[:, start_idx:end_idx] = ds_cell_type_imputed
    
    # Replace any remaining NaN values with zero
    ds_imputed = np.nan_to_num(ds_imputed)
    ds_imputed[ds_imputed < 0] = 0.0
    
    return ds_imputed

def iterative_imputation(ds, cells_per_type=300):
    ds = ds.copy()
    ds[ds == 0] = np.nan
    ds_imputed = np.copy(ds)
    num_genes, num_cells = ds.shape
    num_cell_types = num_cells // cells_per_type

    for i in range(num_cell_types):
        start_idx = i * cells_per_type
        end_idx = start_idx + cells_per_type
        ds_cell_type = ds[:, start_idx:end_idx]
        ds_cell_type_T = ds_cell_type.T
        imputer = IterativeImputer(max_iter=10, random_state=0)
        ds_cell_type_imputed_T = imputer.fit_transform(ds_cell_type_T)
        ds_cell_type_imputed = ds_cell_type_imputed_T.T
        ds_imputed[:, start_idx:end_idx] = ds_cell_type_imputed

    ds_imputed = np.nan_to_num(ds_imputed)
    ds_imputed[ds_imputed < 0] = 0.0
    return ds_imputed

def deep_learning_imputation(ds, cells_per_type=300):
    ds = ds.copy()
    ds[ds == 0] = np.nan
    num_genes, num_cells = ds.shape
    ds_imputed = np.copy(ds)
    num_cell_types = num_cells // cells_per_type

    for i in range(num_cell_types):
        start_idx = i * cells_per_type
        end_idx = start_idx + cells_per_type
        ds_cell_type = ds[:, start_idx:end_idx]
        ds_cell_type_T = ds_cell_type.T
        df = pd.DataFrame(ds_cell_type_T)
        
        # Replace NaNs with zeros for DeepImpute
        df = df.fillna(0)
        
        model = MultiNet()
        model.fit(df)
        imputed_data = model.predict(df)
        ds_cell_type_imputed = imputed_data.to_numpy().T
        ds_imputed[:, start_idx:end_idx] = ds_cell_type_imputed

    # Replace any NaNs or infs with zeros
    ds_imputed = np.nan_to_num(ds_imputed, nan=0.0, posinf=0.0, neginf=0.0)
    ds_imputed[ds_imputed < 0] = 0.0
    return ds_imputed

def graph_convolutional_imputation(ds, adjacency_matrix, num_epochs=100, learning_rate=0.01):
    ds = ds.copy()
    ds[ds == 0] = np.nan
    num_genes, num_cells = ds.shape
    ds_imputed = np.copy(ds)
    
    # Convert adjacency matrix to edge index
    edge_index = np.array(adjacency_matrix.nonzero())
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    # Convert data to PyTorch tensors
    x = torch.tensor(ds_imputed, dtype=torch.float)  # Shape: (num_genes, num_cells)
    
    class GCN(torch.nn.Module):
        def __init__(self, num_features, hidden_channels):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, num_features)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return x

    num_features = num_cells  # Features are cells
    hidden_channels = 64
    
    model = GCN(num_features, hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(x, edge_index)
        mask = torch.isnan(x)
        loss = loss_fn(output[~mask], x[~mask])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    
    # Impute missing values
    model.eval()
    with torch.no_grad():
        imputed = model(x, edge_index)
        x[mask] = imputed[mask]
    ds_imputed = x.numpy()
    ds_imputed[ds_imputed < 0] = 0.0
    return ds_imputed

def graph_diffusion_imputation(ds, adjacency_matrix, alpha=0.5, max_iter=100):
    ds = ds.copy()
    ds[ds == 0] = np.nan
    ds_imputed = np.copy(ds)
    num_genes, num_cells = ds.shape

    degrees = np.sum(adjacency_matrix, axis=1)
    with np.errstate(divide='ignore'):
        D_inv = np.diag(1.0 / degrees)
        D_inv[np.isinf(D_inv)] = 0.0

    P = D_inv.dot(adjacency_matrix)
    P = np.nan_to_num(P)
    I = np.eye(num_genes)
    epsilon = 1e-5
    A = I - alpha * P + epsilon * np.eye(num_genes)

    for cell_idx in range(num_cells):
        y = ds_imputed[:, cell_idx]
        missing_indices = np.isnan(y)
        if np.any(missing_indices):
            x0 = np.zeros(num_genes)
            y_filled = np.nan_to_num(y)
            x, info = bicgstab(A, y_filled, x0=x0, maxiter=max_iter)
            if info != 0:
                print(f"Warning: BiCGSTAB did not converge for cell {cell_idx}, info: {info}")
            y[missing_indices] = x[missing_indices]
            ds_imputed[:, cell_idx] = y

    ds_imputed = np.nan_to_num(ds_imputed)
    ds_imputed[ds_imputed < 0] = 0.0
    return ds_imputed

def saucie_imputation(ds, cells_per_type=300):
    ds = ds.copy()
    num_genes, num_cells = ds.shape
    ds_imputed = np.copy(ds)
    
    num_cell_types = num_cells // cells_per_type

    for i in range(num_cell_types):
        start_idx = i * cells_per_type
        end_idx = start_idx + cells_per_type
        ds_cell_type = ds[:, start_idx:end_idx]
        ds_cell_type_T = ds_cell_type.T

        tf.reset_default_graph()
        saucie = SAUCIE(ds_cell_type_T.shape[1])
        loadtrain = Loader(ds_cell_type_T, shuffle=True)
        saucie.train(loadtrain, steps=1000)
        loadeval = Loader(ds_cell_type_T, shuffle=False)
        rec_ds_T = saucie.get_reconstruction(loadeval)
        rec_ds = rec_ds_T.T
        ds_imputed[:, start_idx:end_idx] = rec_ds

    ds_imputed[ds_imputed < 0] = 0.0
    ds_imputed = np.nan_to_num(ds_imputed)
    return ds_imputed

def magic_imputation(ds, cells_per_type=300):
    ds = ds.copy()
    num_genes, num_cells = ds.shape
    ds_imputed = np.copy(ds)
    
    num_cell_types = num_cells // cells_per_type

    for i in range(num_cell_types):
        start_idx = i * cells_per_type
        end_idx = start_idx + cells_per_type
        ds_cell_type = ds[:, start_idx:end_idx]
        ds_cell_type_T = ds_cell_type.T

        ds_filtered_T = scprep.filter.filter_rare_genes(ds_cell_type_T, min_cells=5)
        ds_normalized_T = scprep.normalize.library_size_normalize(ds_filtered_T)
        ds_sqrt_T = scprep.transform.sqrt(ds_normalized_T)
        magic_operator = magic.MAGIC(
            t='auto',
            n_pca=20,
            n_jobs=-1,
        )
        ds_imputed_T = magic_operator.fit_transform(ds_sqrt_T)
        ds_imputed_cell_type = ds_imputed_T.T

        ds_imputed[:, start_idx:end_idx] = ds_imputed_cell_type

    ds_imputed[ds_imputed < 0] = 0.0
    ds_imputed = np.nan_to_num(ds_imputed)
    return ds_imputed

def scscope_imputation(ds, cells_per_type=300):
    ds = ds.copy()
    ds[ds == 0] = np.nan
    num_genes, num_cells = ds.shape
    ds_imputed = np.copy(ds)
    
    num_cell_types = num_cells // cells_per_type

    for i in range(num_cell_types):
        start_idx = i * cells_per_type
        end_idx = start_idx + cells_per_type
        ds_cell_type = ds[:, start_idx:end_idx]
        ds_cell_type_T = ds_cell_type.T
        DI_model = scScope.train(
            ds_cell_type_T,
            15,
            use_mask=True,
            batch_size=64,
            max_epoch=1000,
            epoch_per_check=100,
            T=2,
            exp_batch_idx_input=[],
            encoder_layers=[],
            decoder_layers=[],
            learning_rate=0.0001,
            beta1=0.05,
            num_gpus=1)
        _, rec_ds_cell_type_T, _ = scScope.predict(ds_cell_type_T, DI_model)
        rec_ds_cell_type = rec_ds_cell_type_T.T
        ds_imputed[:, start_idx:end_idx] = rec_ds_cell_type

    ds_imputed[ds_imputed < 0] = 0.0
    ds_imputed = np.nan_to_num(ds_imputed)
    return ds_imputed

def run_pipeline(imputation_method, method_name, ds_noisy, ds_clean, gt, adjacency_matrix=None, cells_per_type=300):
    print(f"Running {method_name}...")
    if 'Graph' in method_name:
        ds_imputed = imputation_method(ds_noisy, adjacency_matrix)
    else:
        ds_imputed = imputation_method(ds_noisy, cells_per_type=cells_per_type)

    # Evaluate imputation quality
    mse = mean_squared_error(ds_clean.flatten(), ds_imputed.flatten())
    print(f"MSE after {method_name}: {mse:.4f}")

    # Proceed with GENIE3 and ROC AUC evaluation
    ds_imputed_T = ds_imputed.T
    VIM_imputed = GENIE3(ds_imputed_T, nthreads=nthreads, ntrees=100, regulators='all',
                         gene_names=[str(s) for s in range(ds_imputed_T.shape[1])])
    roc_auc = roc_auc_score(gt.flatten(), VIM_imputed.flatten())
    print(f"ROC AUC Score after {method_name}: {roc_auc:.4f}\n")
    return roc_auc, mse

# %%
# Get datasets
datasets = get_datasets()

# Define imputation methods
methods = {
    'KNN Imputation': knn_imputation,
    'Iterative Imputation': iterative_imputation,
    'DeepImpute': deep_learning_imputation,
    'Graph Convolutional Network Imputation': graph_convolutional_imputation,
    'Graph Diffusion Imputation': graph_diffusion_imputation,
    'SAUCIE': saucie_imputation,
    'MAGIC': magic_imputation,
    'scScope': scscope_imputation
}

# Main processing loop
for dataset_info in datasets:
    dataset_id = dataset_info['dataset_id']
    print(f"\nProcessing Dataset {dataset_id}...")
    ds_clean, ds_noisy = load_data(dataset_info)
    if ds_clean is None or ds_noisy is None:
        continue

    num_genes = ds_noisy.shape[0]
    cells_per_type = dataset_info['cells_per_type']
    num_cells = ds_noisy.shape[1]

    target_file = f'../SERGIO/data_sets/{dataset_info["folder_name"]}/Interaction_cID_{dataset_info["dynamics"]}.txt'
    gt = load_ground_truth(target_file, num_genes)
    adjacency_matrix = build_adjacency_matrix(num_genes, target_file)

    # Prepare log file
    log_dir = './results/new_impute'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'Dataset_{dataset_id}_log.txt')
    with open(log_file_path, 'w') as log_file:
        # Evaluate clean data
        print("Evaluating Clean Data...")
        ds_clean_T = ds_clean.T
        VIM_clean = GENIE3(ds_clean_T, nthreads=nthreads, ntrees=100, regulators='all',
                            gene_names=[str(s) for s in range(ds_clean_T.shape[1])])
        roc_auc_clean = roc_auc_score(gt.flatten(), VIM_clean.flatten())
        print(f"ROC AUC Score for Clean Data: {roc_auc_clean:.4f}\n")
        log_file.write(f"ROC AUC Score for Clean Data: {roc_auc_clean:.4f}\n")

        # Evaluate noisy data
        print("Evaluating Noisy Data...")
        ds_noisy_T = ds_noisy.T
        VIM_noisy = GENIE3(ds_noisy_T, nthreads=nthreads, ntrees=100, regulators='all',
                            gene_names=[str(s) for s in range(ds_noisy_T.shape[1])])
        roc_auc_noisy = roc_auc_score(gt.flatten(), VIM_noisy.flatten())
        print(f"ROC AUC Score for Noisy Data: {roc_auc_noisy:.4f}\n")
        log_file.write(f"ROC AUC Score for Noisy Data: {roc_auc_noisy:.4f}\n")

        # Compute MSE between noisy data and clean data
        mse_noisy = mean_squared_error(ds_clean.flatten(), ds_noisy.flatten())
        print(f"MSE between Noisy Data and Clean Data: {mse_noisy:.4f}\n")
        log_file.write(f"MSE between Noisy Data and Clean Data: {mse_noisy:.4f}\n")

        # Run pipeline for each method
        results = {}
        mse_results = {}
        for method_name, method_func in methods.items():
            try:
                if 'Graph' in method_name:
                    roc_auc, mse = run_pipeline(method_func, method_name, ds_noisy, ds_clean, gt, adjacency_matrix, cells_per_type)
                else:
                    roc_auc, mse = run_pipeline(method_func, method_name, ds_noisy, ds_clean, gt, cells_per_type=cells_per_type)
                results[method_name] = roc_auc
                mse_results[method_name] = mse
                log_file.write(f"{method_name} ROC AUC: {roc_auc:.4f}, MSE: {mse:.4f}\n")
            except Exception as e:
                print(f"An error occurred with {method_name}: {e}\n")
                log_file.write(f"An error occurred with {method_name}: {e}\n")

        # Print and log all results
        print("Summary of ROC AUC Scores:")
        print(f"Clean Data: {roc_auc_clean:.4f}")
        print(f"Noisy Data: {roc_auc_noisy:.4f}")
        log_file.write("\nSummary of ROC AUC Scores:\n")
        log_file.write(f"Clean Data: {roc_auc_clean:.4f}\n")
        log_file.write(f"Noisy Data: {roc_auc_noisy:.4f}\n")
        for method_name, roc_auc in results.items():
            print(f"{method_name}: {roc_auc:.4f}")
            log_file.write(f"{method_name}: {roc_auc:.4f}\n")

        print("\nSummary of MSE Values:")
        print(f"Noisy Data: {mse_noisy:.4f}")
        log_file.write("\nSummary of MSE Values:\n")
        log_file.write(f"Noisy Data: {mse_noisy:.4f}\n")
        for method_name, mse in mse_results.items():
            print(f"{method_name}: {mse:.4f}")
            log_file.write(f"{method_name}: {mse:.4f}\n")
