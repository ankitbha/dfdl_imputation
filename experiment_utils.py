import scvi
import anndata
import numpy as np
from utils import gt_benchmark
from GENIE3.GENIE3 import *
from sklearn.metrics import roc_auc_score

def run_genie(x, target_file):
    x[x == 0] = 1e-5
    vim = GENIE3(x, nthreads=12, ntrees=100, regulators='all')
    gt, rescaled_vim = gt_benchmark(vim, target_file)
    roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
    return roc_score

from knn_smoothing.knn_smooth import knn_smoothing
def run_knn(data, k, save_path, it, file_extension='', target_file=None):
    data[data == 0] = 1e-4
    data = np.float64(data)
    data[data == np.nan] = 1e-4
    result = knn_smoothing(data, k)
    np.save(save_path + "y_hat_knn" + file_extension, result)
    roc = run_genie(result, target_file)
    return roc, it  

def run_scvi(data, save_path, it, file_extension='', target_file=None):
    adata = anndata.AnnData(data)
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata)
    model.train()
    x = model.get_normalized_expression(return_numpy=True)
    np.save(save_path + "y_hat_scvi" + file_extension, x)
        
    roc = run_genie(x, target_file)
    return roc, it