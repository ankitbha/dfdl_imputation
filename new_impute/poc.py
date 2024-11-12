import sys, os
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from GENIE3.GENIE3 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from scipy import stats
import SERGIO.SERGIO.sergio as sergio
from sklearn.metrics import roc_auc_score
from copy import deepcopy

nthreads = 12

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

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

def normalized_substitute_dataset(ds):
    ds[ds == 0] = np.nan
    for i in tqdm(range(9)):
        ds_cell_type = ds[:,i*300:(i+1)*300]
        mean_array = np.nanmean(ds_cell_type, axis=1)
        mean_array /= np.nansum(mean_array)
        std_array = np.nanstd(ds_cell_type, axis=1)
        std_array /= np.nansum(mean_array)
        for j in range(100):
            ds_cell_type[j,:] = np.random.normal(loc=mean_array[j], scale=std_array[j], size=300)
    ds[ds<0] = 0.0
    np.nan_to_num(ds, copy=False)
    return(ds)

def novar_substitute_dataset(ds):
    ds[ds == 0] = np.nan
    for i in tqdm(range(9)):
        ds_cell_type = ds[:,i*300:(i+1)*300]
        mean_array = np.nanmean(ds_cell_type, axis=1)
        for j in range(100):
            ds_cell_type[j,:] = np.random.normal(loc=mean_array[j], scale=1.0, size=300)
    ds[ds<0] = 0.0
    np.nan_to_num(ds, copy=False)
    return(ds)

def get_distribution_range(ds, ds_id):
    ds[ds == 0] = np.nan
    for i in tqdm(range(9)):
        ds_cell_type = ds[:,i*300:(i+1)*300]
        mean_array = np.nanmean(ds_cell_type, axis=1)
        std_array = np.nanstd(ds_cell_type, axis=1)
        print(f"DS{ds_id}, Cell {i+1}: Max mean: {np.max(mean_array)}, Min mean: {np.min(mean_array)}")
        print(f"DS{ds_id}, Cell {i+1}: Max std: {np.max(std_array)}, Min std: {np.min(std_array)}")
        fig = plt.figure()
        plt.scatter(mean_array, std_array)
        plt.savefig(f'./results/poc/distribution_range_DS{ds_id}_Cell{i+1}.png')
        plt.close(fig)
    return(0)

def parse_dataset_name(folder_name):
    pattern1 = r'De-noised_(\d+)G_(\d+)T_(\d+)cPerT_dynamics_(\d+)_DS(\d+)'
    pattern2 = r'De-noised_(\d+)G_(\d+)T_(\d+)cPerT_(\d+)_DS(\d+)'
    match_p1 = re.match(pattern1, folder_name)
    match_p2 = re.match(pattern2, folder_name)
    if match_p1:
        return {
            'number_genes': int(match_p1.group(1)),
            'number_bins': int(match_p1.group(2)),
            'number_sc': int(match_p1.group(3)),
            'dynamics': int(match_p1.group(4)),
            'dataset_id': int(match_p1.group(5)),
            "pattern": "De-noised_{number_genes}G_{number_bins}T_{number_sc}cPerT_dynamics_{dynamics}_DS{dataset_id}"
        }
    if match_p2:
        return {
            'number_genes': int(match_p2.group(1)),
            'number_bins': int(match_p2.group(2)),
            'number_sc': int(match_p2.group(3)),
            'dynamics': int(match_p2.group(4)),
            'dataset_id': int(match_p2.group(5)),
            "pattern": "De-noised_{number_genes}G_{number_bins}T_{number_sc}cPerT_{dynamics}_DS{dataset_id}"
        }
    return

def get_datasets():
    datasets = []
    for folder_name in os.listdir('../SERGIO/data_sets'):
        dataset_info = parse_dataset_name(folder_name)
        if dataset_info:
            datasets.append(dataset_info)
    return sorted(datasets, key=lambda x: x['dataset_id'])

def process_dataset(dataset_info, clean_path):
    dataset_id = dataset_info['dataset_id']
    dataset_name = f"DS{dataset_id}"
    
    ds_clean = np.load(clean_path)
    target_file = f"../SERGIO/data_sets/{dataset_info['pattern'].format(**dataset_info)}/Interaction_cID_{dataset_info['dynamics']}.txt"

    gt = np.zeros((dataset_info['number_genes'], dataset_info['number_genes']))
    with open(target_file, 'r') as f:
        lines = f.readlines()
        f.close()
        for j in range(len(lines)):
            line = lines[j]
            line_list = line.split(',')
            target_index = int(float(line_list[0]))
            num_regs = int(float(line_list[1]))
            for i in range(num_regs):
                try:
                    reg_index = int(float(line_list[i+2]))
                    gt[reg_index,target_index] = 1
                except:
                    continue

    for method_name, method_func in [
        ("substitute", substitute_dataset),
        ("normalized_substitute", normalized_substitute_dataset),
        ("novar_substitute", novar_substitute_dataset)
    ]:
        start_time = time.time()
        ds_processed = method_func(ds_clean.astype(np.float32))
        VIM = GENIE3(np.transpose(ds_processed), nthreads=nthreads, ntrees=100, regulators='all',
                     gene_names=[str(s) for s in range(np.transpose(ds_processed).shape[1])])
        
        roc_auc = roc_auc_score(gt.flatten(), VIM.flatten())
        elapsed_time = time.time() - start_time
        
        with open(f'./results/poc/log_DS{dataset_id}.txt', 'a') as log_file:
            log_file.write(f"Dataset: {dataset_name}, Method: {method_name}\n")
            log_file.write(f"ROC AUC Score: {roc_auc}\n")
            log_file.write(f"Elapsed time: {elapsed_time:.2f} seconds\n\n")

    get_distribution_range(ds_clean.astype(float32), dataset_id)

ensure_dir('./results/poc')

cleans = {
    1: '../SERGIO/imputation_data_2/DS1/DS6_clean.npy',
    2: '../SERGIO/imputation_data_2/DS2/DS6_clean.npy',
    3: '../SERGIO/imputation_data_2/DS3/DS6_clean.npy'
}

noisy = {
    1: '../SERGIO/imputation_data_2/DS1/DS6_45_iter_0.npy',
    2: '../SERGIO/imputation_data_2/DS2/DS6_45_iter_0.npy',
    3: '../SERGIO/imputation_data_2/DS3/DS6_45_iter_0.npy'
}

datasets = [get_datasets()[1]]
for dataset_info in tqdm(datasets):
    process_dataset(dataset_info, noisy[dataset_info['dataset_id']])

print("Analysis complete. Results saved in ./results/poc/")