from GENIE3.GENIE3 import *
import sys, os
import numpy as np
import pandas as pd
import scprep
import matplotlib
import json
import importlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

#sys.path.append(os.getcwd())

# path_to_SERGIO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SERGIO'))
# path_to_MAGIC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MAGIC'))
# path_to_SAUCIE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SAUCIE'))
# path_to_scScope = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scScope'))
# path_to_DeepImpute = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deepimpute'))

# if path_to_SERGIO not in sys.path:
#     sys.path.insert(0, path_to_SERGIO)
from SERGIO.SERGIO.sergio import sergio
# if path_to_MAGIC not in sys.path:
#     sys.path.insert(0, path_to_MAGIC)
import MAGIC.magic as magic
# if path_to_SAUCIE not in sys.path:
#     sys.path.insert(0, path_to_SAUCIE)
import SAUCIE.SAUCIE as SAUCIE
# if path_to_scScope not in sys.path:
#     sys.path.insert(0, path_to_scScope)
import scScope.scscope.scscope as DeepImpute
# if path_to_DeepImpute not in sys.path:
#     sys.path.insert(0, path_to_DeepImpute)
import deepimpute.deepimpute as deepimpute

from utils import gt_benchmark, reload_modules, delete_modules 
from utils import plot_precisions, precision_at_k, flat_precision_at_k

def run_sergio(input_file, reg_file, ind):
    # Run SERGIO
    n_genes = 1200
    if ind == 1:
        n_genes = 100
    if ind == 2:
        n_genes = 400
    sim = sergio(
        number_genes=n_genes, 
        number_bins = 9, 
        number_sc = 300,
        # In paper
        noise_params = 1,
        # In paper
        decays=0.8, 
        sampling_state=15, 
        noise_type='dpd')
    sim.build_graph(input_file_taregts=input_file, input_file_regs=reg_file, shared_coop_state=2)
    sim.simulate()
    
    # Get Expression Data
    expr = sim.getExpressions()
    expr_clean = np.concatenate(expr, axis = 1)
    ds_str = 'DS' + str(ind)
    save_path = './imputations/' + ds_str
    
    # Save simulated data variants
    np.save(save_path + '/DS6_clean', expr_clean)
    np.save(save_path + '/DS6_expr', expr)
    cmat_clean = sim.convert_to_UMIcounts(expr)
    cmat_clean = np.concatenate(cmat_clean, axis = 1)
    np.save(save_path + '/DS6_clean_counts', cmat_clean)

    # Add Technical Noise - Steady State Simulations
    expr_O = sim.outlier_effect(expr, outlier_prob = 0.01, mean = 5, scale = 1)
    libFactor, expr_O_L = sim.lib_size_effect(expr_O, mean = 4.5, scale = 0.7)
    binary_ind = sim.dropout_indicator(expr_O_L, shape = 8, percentile = 45)
    expr_O_L_D = np.multiply(binary_ind, expr_O_L)
    count_matrix = sim.convert_to_UMIcounts(expr_O_L_D)
    count_matrix = np.concatenate(count_matrix, axis = 1)
    np.save(save_path + '/DS6_45', count_matrix)

def run_saucie(x_path, y_path, ind):
    #reload_modules('tensorflow.compat')
    tf = importlib.import_module('tensorflow.compat.v1')
    #importlib.reload(SAUCIE)
    tf.disable_v2_behavior()
    ds_str = 'DS' + str(ind)
    save_path = './imputations/' + ds_str
    print("loading data")
    y = np.transpose(np.load(y_path))
    x = np.transpose(np.load(x_path))
    print("reset graph")
    tf.reset_default_graph()
    print("Initialize saucie")
    saucie = SAUCIE.SAUCIE(y.shape[1])
    print("Load saucie")
    loadtrain = SAUCIE.Loader(y, shuffle=True)
    print("Train saucie")
    saucie.train(loadtrain, steps=1000)

    loadeval = SAUCIE.Loader(y, shuffle=False)
    # embedding = saucie.get_embedding(loadeval)
    # number_of_clusters, clusters = saucie.get_clusters(loadeval)
    rec_y = saucie.get_reconstruction(loadeval)
    save_str = '/yhat_SAUCIE'
    np.save(save_path + save_str, rec_y)

def run_deepImpute(x_path, y_path, ind):
    #reload_modules('tensorflow.compat')
    importlib.invalidate_caches()
    multinet = importlib.import_module('deepimpute.deepimpute.multinet')
    importlib.reload(multinet)
    tf = importlib.import_module('tensorflow.compat.v1')
    #tf = importlib.import_module('tensorflow')
    tf.init_scope()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    ds_str = 'DS' + str(ind)
    save_path = './imputations/' + ds_str
    y = np.transpose(np.load(y_path))
    y = pd.DataFrame(y)
    x = np.transpose(np.load(x_path))
    x = pd.DataFrame(x)
    multinet = multinet.MultiNet()
    multinet.fit(y,cell_subset=1,minVMR=0.5)
    imputedData = multinet.predict(y)
    yhat_deepimpute = imputedData.to_numpy()
    save_str = '/yhat_deepImpute'
    np.save(save_path + save_str, yhat_deepimpute)

def run_magic(x_path, y_path, ind):
    ds_str = 'DS' + str(ind)
    save_path = './imputations/' + ds_str
    print(x_path, y_path)
    y = np.transpose(np.load(y_path))
    x = np.transpose(np.load(x_path))
    
    y_hat = scprep.filter.filter_rare_genes(y, min_cells=5)
    y_norm = scprep.normalize.library_size_normalize(y_hat)
    y_norm = scprep.transform.sqrt(y_norm)

    for t_val in [2, 7, 'auto']:
        magic_op = magic.MAGIC(
            # knn=5,
            # knn_max=None,
            # decay=1,
            # Variable changed in paper
            t=t_val,
            n_pca=20,
            # solver="exact",
            # knn_dist="euclidean",
            n_jobs=-1,
            # random_state=None,
            # verbose=1,
        )
        y_hat = magic_op.fit_transform(y_norm, genes='all_genes')
        save_str = '/yhat_MAGIC_t_' + str(t_val)
        np.save(save_path + save_str, y_hat)

def run_scScope(x_path, y_path, ind):
    ds_str = 'DS' + str(ind)
    save_path = './imputations/' + ds_str
    y = np.transpose(np.load(y_path))
    x = np.transpose(np.load(x_path))
    DI_model = DeepImpute.train(
          y,
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
    latent_code, rec_y, _ = DeepImpute.predict(y, DI_model, batch_effect=[])
    save_str = '/yhat_scScope'
    np.save(save_path + save_str, rec_y)

def run_simulations(datasets, sergio=True, saucie=True, scScope=True, deepImpute=True, magic=True, genie=True, roc=True, precision_recall_k=True):
    target_path = ''
    regs_path = ''
    results = {}
    count_methods = 2
    for i in tqdm(datasets):
        individual_results = {}
        if i == 1:   
            target_path = target_file = './SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt'
            regs_path = './SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt'
        elif i == 2:
            target_path = target_file = './SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/Interaction_cID_5.txt'
            regs_path = './SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/Regs_cID_5.txt'
        else:
            target_path = target_file = './SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Interaction_cID_6.txt'
            regs_path = './SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Regs_cID_6.txt'
        ds_str = 'DS' + str(i)
        save_path = './imputations/' + ds_str

        if sergio:
            print(f"---> Running SERGIO on DS{i}")
            run_sergio(target_path, regs_path, i)
            count_methods += 1
        
        if saucie:
            print(f"---> Running SAUCIE on DS{i}")
            run_saucie(save_path + '/DS6_clean.npy', save_path + '/DS6_45.npy', i)
            count_methods += 1
        
        if scScope:
            print(f"---> Running scScope on DS{i}")
            run_scScope(save_path + '/DS6_clean.npy', save_path + '/DS6_45.npy', i)
            count_methods += 1

        if deepImpute:
            print(f"---> Running DeepImpute on DS{i}")
            run_deepImpute(save_path + '/DS6_clean.npy', save_path + '/DS6_45.npy', i)
            count_methods += 1

        if magic:
            print(f"---> Running MAGIC on DS{i}")
            run_magic(save_path + '/DS6_clean.npy', save_path + '/DS6_45.npy', i)
            count_methods += 3

        y = np.transpose(np.load(save_path + '/DS6_45.npy'))
        x = np.transpose(np.load(save_path + '/DS6_clean.npy'))
        
        if genie:
            if sergio:
            # Run GENIE3 on Clean Data
                print(f"---> Running GENIE3 on Clean Data for DS{i}")
                VIM_CLEAN = GENIE3(x, nthreads=12, ntrees=100)        
                gt, rescaled_vim = gt_benchmark(VIM_CLEAN, target_file)
                if roc:
                    roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                    individual_results['Clean ROC_AUC'] = float('%.2f'%(roc_score))
                if precision_recall_k:
                    k = range(1, gt.shape[0])
                    k_flat = range(1, gt.size)
                    precision_k = precision_at_k(gt, rescaled_vim, k)
                    individual_results['Clean Precision@k'] = precision_k

                # Run GENIE3 on Noisy Data
                print(f"---> Running GENIE3 on Noisy Data for DS{i}")
                VIM_NOISY = GENIE3(y, nthreads=12, ntrees=100)        
                gt, rescaled_vim = gt_benchmark(VIM_NOISY, target_file)
                if roc:
                    roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                    individual_results['Noisy ROC_AUC'] = float('%.2f'%(roc_score))
                if precision_recall_k:
                    k = range(1, gt.shape[0])
                    k_flat = range(1, gt.size)
                    precision_k = precision_at_k(gt, rescaled_vim, k)
                    individual_results['Noisy Precision@k'] = precision_k

            # Run GENIE3 on SAUCIE Data
            if saucie:
                y_hat_saucie = np.load(save_path + '/yhat_SAUCIE.npy')

                print(f"---> Running GENIE3 on SAUCIE Data for DS{i}")
                VIM_SAUCIE = GENIE3(y_hat_saucie, nthreads=12, ntrees=100)
                gt, rescaled_vim = gt_benchmark(VIM_SAUCIE, target_file)
                # np.save(save_path + '/VIM_SAUCIE.npy', rescaled_vim)
                # np.save(save_path + '/gt_SAUCIE.npy', gt)
                if roc:
                    roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                    individual_results['SAUCIE ROC_AUC'] = float('%.2f'%(roc_score))
                if precision_recall_k:
                    k = range(1, gt.shape[0])
                    k_flat = range(1, gt.size)
                    precision_k = precision_at_k(gt, rescaled_vim, k)
                    individual_results['SAUCIE Precision@k'] = precision_k
            
            # Run GENIE3 on scScope Data
            if scScope:
                y_hat_scscope = np.load(save_path + '/yhat_scScope.npy')
                y_hat_scScope  = y_hat_scscope.copy()
                y_hat_scScope[y_hat_scScope == 0] = 1e-5

                print(f"---> Running GENIE3 on scScope Data for DS{i}")
                VIM_scScope = GENIE3(y_hat_scScope, nthreads=12, ntrees=100)
                gt, rescaled_vim = gt_benchmark(VIM_scScope, target_file)
                if roc:
                    roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                    individual_results['scScope ROC_AUC'] = float('%.2f'%(roc_score))
                if precision_recall_k:
                    k = range(1, gt.shape[0])
                    k_flat = range(1, gt.size)
                    precision_k = precision_at_k(gt, rescaled_vim, k)
                    individual_results['scScope Precision@k'] = precision_k

            # Run GENIE3 on DeepImpute Data
            if deepImpute:
                y_hat_deepImpute = np.load(save_path + '/yhat_deepImpute.npy')

                print(f"---> Running GENIE3 on DeepImpute Data for DS{i}")
                VIM_deepImpute = GENIE3(y_hat_deepImpute, nthreads=12, ntrees=100)
                gt, rescaled_vim = gt_benchmark(VIM_deepImpute, target_file)
                np.save(save_path + '/VIM_deepImpute.npy', rescaled_vim)
                np.save(save_path + '/gt_deepImpute.npy', gt)
                print("saved deepimpute files")
                if roc:
                    roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                    individual_results['DeepImpute ROC_AUC'] = float('%.2f'%(roc_score))
                if precision_recall_k:
                    k = range(1, gt.shape[0])
                    k_flat = range(1, gt.size)
                    precision_k = precision_at_k(gt, rescaled_vim, k)
                    individual_results['DeepImpute Precision@k'] = precision_k

            # Run GENIE3 on MAGIC Data
            if magic:
                y_hat_magic_t2 = np.load(save_path + '/yhat_MAGIC_t_2.npy')
                y_hat_magic_t7 = np.load(save_path + '/yhat_MAGIC_t_7.npy')
                y_hat_magic_t_auto = np.load(save_path + '/yhat_MAGIC_t_auto.npy')

                print(f"---> Running GENIE3 on MAGIC t=2 for DS{i}")
                VIM_MAGIC = GENIE3(y_hat_magic_t2, nthreads=12, ntrees=100)
                gt, rescaled_vim = gt_benchmark(VIM_MAGIC, target_file)
                if roc:
                    roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                    individual_results['MAGIC t=2 ROC_AUC'] = float('%.2f'%(roc_score))
                if precision_recall_k:
                    k = range(1, gt.shape[0])
                    k_flat = range(1, gt.size)
                    precision_k = precision_at_k(gt, rescaled_vim, k)
                    individual_results['MAGIC t=2 Precision@k'] = precision_k

                print(f"---> Running GENIE3 on MAGIC t=7 for DS{i}")
                VIM_MAGIC = GENIE3(y_hat_magic_t7, nthreads=12, ntrees=100)
                gt, rescaled_vim = gt_benchmark(VIM_MAGIC, target_file)
                if roc:
                    roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                    individual_results['MAGIC t=7 ROC_AUC'] = float('%.2f'%(roc_score))
                if precision_recall_k:
                    k = range(1, gt.shape[0])
                    k_flat = range(1, gt.size)
                    precision_k = precision_at_k(gt, rescaled_vim, k)
                    individual_results['MAGIC t=7 Precision@k'] = precision_k
                
                print(f"---> Running GENIE3 on MAGIC t=default for DS{i}")
                VIM_MAGIC = GENIE3(y_hat_magic_t_auto, nthreads=12, ntrees=100)
                gt, rescaled_vim = gt_benchmark(VIM_MAGIC, target_file)
                if roc:
                    roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                    individual_results['MAGIC t=default ROC_AUC'] = float('%.2f'%(roc_score))
                if precision_recall_k:
                    k = range(1, gt.shape[0])
                    k_flat = range(1, gt.size)
                    precision_k = precision_at_k(gt, rescaled_vim, k)
                    individual_results['MAGIC t=default Precision@k'] = precision_k

        print(individual_results)
        results[ds_str] = individual_results
        # write individual results to JSON file
        with open(save_path + '/precision_recall_data.json', 'w') as fp:
            json.dump(individual_results, fp)
    return results#, count_methods

def create_correlation_plots(datasets):
    for i in tqdm(datasets):
        print(f"---> Calculating correlations for data from DS{i}")
        ds_str = 'DS' + str(i)
        save_path = './imputations/' + ds_str

        # Load saved data
        y = np.transpose(np.load(save_path + '/DS6_45.npy'))
        x = np.transpose(np.load(save_path + '/DS6_clean.npy'))
        
        # Load MAGIC imputed data
        y_hat_magic_t2 = np.load(save_path + '/yhat_MAGIC_t_2.npy')
        y_hat_magic_t7 = np.load(save_path + '/yhat_MAGIC_t_7.npy')
        y_hat_magic_t_auto = np.load(save_path + '/yhat_MAGIC_t_auto.npy')

        # Create correlation dataframes
        x_corr = pd.DataFrame(x).corr()
        y_corr = pd.DataFrame(y).corr()
        t2_corr = pd.DataFrame(y_hat_magic_t2).corr()
        t7_corr = pd.DataFrame(y_hat_magic_t7).corr()
        t_auto_corr = pd.DataFrame(y_hat_magic_t_auto).corr()
        
        # Create subplots
        fig, axs = plt.subplots(3, 2, figsize=(12, 12))
        fig.suptitle(f'Correlation plots for DS{i}')
        axs[0, 0].stairs(*np.histogram(x_corr, bins = 100, density=True), fill=True, color="green")
        axs[0, 0].set_xlim(-1, 1)
        axs[0, 0].set_title("Clean")

        axs[0, 1].stairs(*np.histogram(y_corr, bins = 100, density=True), fill=True, color="green")
        axs[0, 1].set_xlim(-1, 1)
        axs[0, 1].set_title("Noisy")

        axs[1, 0].stairs(*np.histogram(t2_corr, bins = 100, density=True), fill=True, color="green")
        axs[1, 0].set_xlim(-1, 1)
        axs[1, 0].set_title("Imputed; t=2")

        axs[1, 1].stairs(*np.histogram(t7_corr, bins = 100, density=True), fill=True, color="green")
        axs[1, 1].set_xlim(-1, 1)
        axs[1, 1].set_title("Imputed; t=7")

        axs[2, 0].stairs(*np.histogram(t_auto_corr, bins = 100, density=True), fill=True, color="green")
        axs[2, 0].set_xlim(-1, 1)
        axs[2, 0].set_title("Imputed; t=default")

        for ax in axs.flat:
            ax.set(xlabel = 'Corr. Coeff.', ylabel="Density (%)")
        # for ax in axs.flat:
        #     ax.label_outer()
        fig.tight_layout(pad=2.0)
        plt.show()
    return