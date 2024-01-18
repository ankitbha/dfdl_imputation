import numpy as np
import pandas as pd
import sys
from IPython import get_ipython
from tqdm import tqdm
import matplotlib.pyplot as plt

def delete_modules(modname):
    from sys import modules
    items = modules.copy().items()
    for key, mod in items:
        try:
            try:
                if modname in str(mod):
                    # print("val", key, mod)
                    del modules[key]
            except:
                pass            
        except AttributeError:
            pass

def reload_modules(modname):
    ipython = get_ipython()
    delete_modules(modname)
    if '__IPYTHON__' in globals():
        ipython.magic('load_ext autoreload')
        ipython.magic('autoreload 2')

def gt_benchmark(virtual_imputation, target_file):
    # Create numpy array of same size as imputation_dataset
    gt_temp = np.zeros_like(virtual_imputation)
    f = open(target_file,'r')
    Lines = f.readlines()
    f.close()
    # For each real gene and measured gene expressions, set new array at coordinates to 1
    for j in tqdm(range(len(Lines))):
        line = Lines[j]
        line_list = line.split(',')
        target_index = int(float(line_list[0]))
        num_regs = int(float(line_list[1]))
        # skip if gene is not present in filtered dataset
        if target_index >= gt_temp.shape[1]:
            for i in range(0, target_index + 1 - gt_temp.shape[1]):
                new_column = np.zeros((gt_temp.shape[0], 1), dtype=int)
                gt_temp = np.append(gt_temp, new_column, axis=1)
                virtual_imputation = np.append(virtual_imputation, new_column, axis=1)
        for i in range(num_regs):
            reg_index = int(float(line_list[i+2]))
            gt_temp[reg_index,target_index] = 1  
    return gt_temp, virtual_imputation

def precision_at_k(vim, gt, k_vals, get_locs=False):
    k_averages = []
    i = len(vim) - 1
    nested_precisions = []
    avgs = []
    for index in tqdm(vim):
        gt_row = gt[i]
        sorted = np.argsort(vim[i])
        gt_sum = np.sum(gt_row)
        precisions = [0] * len(k_vals)
        i -= 1
        for k in k_vals[::-1]:
            top_k = sorted[:k]
            top_k_gt = gt_row[top_k]
            if k != k_vals[-1]:
                gt_sum -= top_k_gt[-1]
            precisions[k - 1] = gt_sum / k
        nested_precisions.append(precisions)
    nested_precisions = np.array(nested_precisions)
    for i, val in enumerate(nested_precisions[0]):
        avg = np.mean(nested_precisions[:,i])
        avgs.append(avg)
    return avgs

def flat_precision_at_k(vim, gt, k_vals, get_locs=False):
    vim_flat = vim.flatten()
    gt_flat = gt.flatten()
    sorted = np.argsort(vim_flat)
    precisions = [0] * gt.size
    gt_sum = 0
    for k in tqdm(k_vals[::-1]):
        top_k = sorted[:k]
        if k == k_vals[-1]:  
            gt_sum = np.sum(gt_flat[top_k])
        else:
            gt_sum -= gt_flat[top_k[-1]]
        precisions[k - 1] = gt_sum / k
    avg_precisions = [0] * len(k_vals)
    total_relevant = np.sum(gt_flat)
    last_dot = np.dot(precisions, gt_flat)
    i = len(k_vals) - 1
    for index, precision in enumerate(tqdm(precisions)):
        if index == 0:
            pass
        else:
            total_relevant -= gt_flat[i]
        last_dot -= (precisions[i] * gt_flat[i])
        if total_relevant != 0:
            avg_precisions[i] = (1 / total_relevant) * last_dot
        else:
            avg_precisions[i] = 0
        i -= 1
    return precisions[::gt.shape[0]], avg_precisions[::gt.shape[0]]

def plot_precisions(precisions, flat_precisions, avg_precisions, method_name, gt=None):
    plt.figure(figsize=(15,8))
    plt.plot(pd.DataFrame(precisions), color='orange', label='Precision @ k')
    plt.plot(pd.DataFrame(flat_precisions), color='green', label='Precision @ k Flat')
    plt.plot(pd.DataFrame(avg_precisions), color='blue', label='Avg. Precision @ K Flat')
    if gt is not None:
        plt.axhline(gt.sum() / gt.size, linestyle='dashed', color='grey', label='% of GT == 1')
    plt.legend()
    plt.xlabel("K")
    plt.ylabel("Precision")
    plt.title(f"{method_name} Precision @ K Curves")
    plt.show()