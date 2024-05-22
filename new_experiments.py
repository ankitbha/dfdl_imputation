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

from Pearson.pearson import Pearson

import arboreto as arboreto

from utils import gt_benchmark, reload_modules, delete_modules 
from utils import plot_precisions, precision_at_k

def run_sergio(input_file, reg_file, ind, n_genes=1200, n_bins=9, n_sc=300, file_extension = ''):
    # Run SERGIO
    if ind == 1:
        n_genes = 100
    if ind == 2:
        n_genes = 400
    sim = sergio(
        number_genes=n_genes, 
        number_bins = n_bins, 
        number_sc = n_sc,
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
    save_path = './zero_imputation_experiments/' + ds_str
    
    # Save simulated data variants
    np.save(save_path + '/DS6_clean', expr_clean )
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
    np.save(save_path + '/DS6_noisy', count_matrix)

def run_saucie(x_path, y_path, ind, file_extension = ''):
    #reload_modules('tensorflow.compat')
    tf = importlib.import_module('tensorflow.compat.v1')
    #importlib.reload(SAUCIE)
    tf.disable_v2_behavior()
    ds_str = 'DS' + str(ind)
    save_path = './zero_imputation_experiments/' + ds_str
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
    save_str = '/yhat_SAUCIE'+ file_extension
    np.save(save_path + save_str, rec_y)

def run_deepImpute(x_path, y_path, ind, file_extension = ''):
    #reload_modules('tensorflow.compat')
    importlib.invalidate_caches()
    multinet = importlib.import_module('deepimpute.deepimpute.multinet')
    importlib.reload(multinet)
    tf = importlib.import_module('tensorflow.compat.v1')
    #tf = importlib.import_module('tensorflow')
    tf.init_scope()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    ds_str = 'DS' + str(ind)
    save_path = './zero_imputation_experiments/' + ds_str
    y = np.transpose(np.load(y_path))
    y = pd.DataFrame(y)
    x = np.transpose(np.load(x_path))
    x = pd.DataFrame(x)
    multinet = multinet.MultiNet()
    multinet.fit(y,cell_subset=1,minVMR=0.5)
    imputedData = multinet.predict(y)
    yhat_deepimpute = imputedData.to_numpy()
    save_str = '/yhat_deepImpute' + file_extension
    np.save(save_path + save_str, yhat_deepimpute)

def run_magic(x_path, y_path, ind, file_extension = ''):
    ds_str = 'DS' + str(ind)
    save_path = './zero_imputation_experiments/' + ds_str
    print(x_path, y_path)
    y = np.transpose(np.load(y_path))
    x = np.transpose(np.load(x_path))
    
    y_hat = scprep.filter.filter_rare_genes(y, min_cells=5)
    y_norm = scprep.normalize.library_size_normalize(y_hat)
    y_norm = scprep.transform.sqrt(y_norm)

    for t_val in [7]:#[2, 7, 'auto']:
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
        save_str = '/yhat_MAGIC_t_' + str(t_val) + file_extension
        np.save(save_path + save_str, y_hat)

def run_scScope(x_path, y_path, ind, file_extension = ''):
    ds_str = 'DS' + str(ind)
    save_path = './zero_imputation_experiments/' + ds_str
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
    save_str = '/yhat_scScope' + file_extension
    np.save(save_path + save_str, rec_y)

def run_arboreto(path, roc, precision_recall_k, method_name, target, ind, regs=None, file_extension = ''):
    if regs is None:
        regs = 'all'
    dataset = np.transpose(np.load(path))
    df = pd.DataFrame(dataset)
    c_names = [str(c) for c in df.columns]
    df.columns = c_names
    from arboreto import algo
    #network = algo.grnboost2(expression_data=df, tf_names=regs, verbose=True)
    network = algo.genie3(expression_data=df, tf_names=regs, verbose=True)
    network['TF'] = network['TF'].astype(int)
    network['target'] = network['target'].astype(int)
    num_rows = network['target'].max() + 1
    num_cols = network['TF'].max() + 1
    matrix = np.zeros((num_rows, num_cols))
    ret_dict = {}
    for _, row in network.iterrows():
        matrix[int(row['target']), int(row['TF'])] = row['importance']
    
    gt, rescaled_vim = gt_benchmark(matrix, target)
    if roc:
        roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
        ret_dict['DS' + str(ind) + ' arboreto ' + method_name + ' ROC_AUC'] = float('%.2f'%(roc_score))
    if precision_recall_k:
        k = range(1, gt.size)
        precision_k = precision_at_k(gt, rescaled_vim, k)
        ret_dict['DS' + str(ind) + ' arboreto ' + method_name + ' Precision@k'] = precision_k
    
    return ret_dict

def run_pearson(path, target, roc, precision_recall_k, method_name, ind, file_extension = ''):
    dataset = np.transpose(np.load(path))
    pearson = Pearson(dataset, '').values
    gt, rescaled_vim = gt_benchmark(pearson, target)
    ret_dict = {}
    if roc:
        roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
        ret_dict['DS' + str(ind) + ' Pearson ' + method_name + ' ROC_AUC'] = float('%.2f'%(roc_score))
    if precision_recall_k:
        k = range(1, gt.size)
        precision_k = precision_at_k(gt, rescaled_vim, k)
        ret_dict['DS' + str(ind) + ' Pearson ' + method_name + ' Precision@K'] = precision_k
    print(ret_dict)
    return ret_dict

def run_simulation(dataset, sergio=True, saucie=True, scScope=True, deepImpute=True, magic=True, genie=True, pearson=False, arboreto=True, roc=True, precision_recall_k=True, run_with_regs=False, iteration=0):
    target_file = ''
    regs_path = ''
    results = {}
    count_methods = 2
    i = dataset
    individual_results = {}
    file_extension = f"_iter_{iteration}"
    if dataset == 1:   
        target_file = './SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt'
        regs_path = './SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt'
    elif dataset == 2:
        target_file = './SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/Interaction_cID_5.txt'
        regs_path = './SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/Regs_cID_5.txt'
    else:
        target_file = './SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Interaction_cID_6.txt'
        regs_path = './SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Regs_cID_6.txt'
    ds_str = 'DS' + str(dataset)
    save_path = './zero_imputation_experiments/' + ds_str

    if sergio:
        print(f"---> Running SERGIO on DS{dataset}")
        run_sergio(target_file, regs_path, ind=dataset, file_extension=file_extension)
        count_methods += 1
    
    if saucie:
        print(f"---> Running SAUCIE on DS{dataset}")
        run_saucie(save_path + '/DS6_clean.npy', save_path + '/DS6_noisy.npy', dataset, file_extension)
        count_methods += 1
    
    if scScope:
        print(f"---> Running scScope on DS{dataset} ")
        run_scScope(save_path + '/DS6_clean.npy', save_path + '/DS6_noisy.npy', i, file_extension)
        count_methods += 1

    if deepImpute:
        print(f"---> Running DeepImpute on DS{dataset} ")
        run_deepImpute(save_path + '/DS6_clean.npy', save_path + '/DS6_noisy.npy', i, file_extension)
        count_methods += 1

    if magic:
        print(f"---> Running MAGIC on DS{dataset} ")
        run_magic(save_path + '/DS6_clean.npy', save_path + '/DS6_noisy.npy', i, file_extension)
        count_methods += 3

    y = np.transpose(np.load(save_path + '/DS6_noisy.npy'))
    x = np.transpose(np.load(save_path + '/DS6_clean.npy'))

    if arboreto:
        reg_file = None
        if dataset== 1:
            reg_file = './SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt'
        elif dataset== 2:
            reg_file = './SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/Regs_cID_5.txt'
        else:
            reg_file = 'SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Regs_cID_6.txt'
        master_regs = pd.read_table(reg_file, header=None, sep=',')
        master_regs = master_regs[0].values.astype(int).astype(str).tolist()

        regulators = []
        regulator_file = open(target_file, 'r')
        lines = regulator_file.readlines()
        for line in lines:
            row = line.split(',')
            num_regs_row = int(float(row[1]))
            if num_regs_row != 0:
                for i in range(2, num_regs_row + 2):
                    regulators.append(str(int(float(row[i]))))
        regs = list(set(regulators))
        regs = [i for i in regs if i not in master_regs]

        # if sergio:
        print(f"---> Running arboreto on Clean Data for DS{dataset} ")
        arboreto_results = run_arboreto(save_path + '/DS6_clean.npy', roc, precision_recall_k, 'Clean', target_file, i, regs)
        individual_results.update(arboreto_results)
        print(f"---> Running arboreto on Noisy Data for DS{dataset} ")
        arboreto_results = run_arboreto(save_path + '/DS6_noisy.npy', roc, precision_recall_k, 'Noisy', target_file, i, regs)
        individual_results.update(arboreto_results)
        if saucie:
            print(f"---> Running arboreto on SAUCIE Data for DS{dataset} ")
            arboreto_results = run_arboreto(save_path + '/yhat_SAUCIE.npy', roc, precision_recall_k, 'SAUCIE', target_file, i, regs)
            individual_results.update(arboreto_results)
        if scScope:
            print(f"---> Running arboreto on scScope Data for DS{dataset} ")
            arboreto_results = run_arboreto(save_path + '/yhat_scScope.npy', roc, precision_recall_k, 'scScope', target_file, i, regs)
            individual_results.update(arboreto_results)
        if deepImpute:
            print(f"---> Running arboreto on DeepImpute Data for DS{dataset} ")
            arboreto_results = run_arboreto(save_path + '/yhat_deepImpute.npy', roc, precision_recall_k, 'DeepImpute', target_file, i, regs)
            individual_results.update(arboreto_results)
        if magic:
            print(f"---> Running arboreto on MAGIC t=2 Data for DS{dataset} ")
            arboreto_results = run_arboreto(save_path + '/yhat_MAGIC_t_2.npy', roc, precision_recall_k, 'MAGIC t=2', target_file, i, regs)
            individual_results.update(arboreto_results)
            print(f"---> Running arboreto on MAGIC t=7 Data for DS{dataset} ")
            arboreto_results = run_arboreto(save_path + '/yhat_MAGIC_t_7.npy', roc, precision_recall_k, 'MAGIC t=7', target_file, i, regs)
            individual_results.update(arboreto_results)
            print(f"---> Running arboreto on MAGIC t=default Data for DS{dataset} ")
            arboreto_results = run_arboreto(save_path + '/yhat_MAGIC_t_auto.npy', roc, precision_recall_k, 'MAGIC t=default', target_file, i, regs)
            individual_results.update(arboreto_results)
    
    if pearson:
        print(f"---> Running Pearson on Clean Data for DS{dataset} ")
        pearson_results = run_pearson(save_path + '/DS6_clean.npy', target_file, roc, precision_recall_k, 'Clean', i)
        individual_results.update(pearson_results)
        print(f"---> Running Pearson on Noisy Data for DS{dataset} ")
        pearson_results = run_pearson(save_path + '/DS6_noisy.npy', target_file, roc, precision_recall_k, 'Noisy', i)
        individual_results.update(pearson_results)
        if saucie:
            print(f"---> Running Pearson on SAUCIE Data for DS{dataset} ")
            pearson_results = run_pearson(save_path + '/yhat_SAUCIE.npy', target_file, roc, precision_recall_k, 'SAUCIE', i)
            individual_results.update(pearson_results)
        if scScope:
            print(f"---> Running Pearson on scScope Data for DS{dataset} ")
            pearson_results = run_pearson(save_path + '/yhat_scScope.npy', target_file, roc, precision_recall_k, 'scScope', i)
            individual_results.update(pearson_results)
        if deepImpute:
            print(f"---> Running Pearson on DeepImpute Data for DS{dataset} ")
            pearson_results = run_pearson(save_path + '/yhat_deepImpute.npy', target_file, roc, precision_recall_k, 'DeepImpute', i)
            individual_results.update(pearson_results)
        if magic:
            print(f"---> Running Pearson on MAGIC t=2 Data for DS{dataset} ")
            pearson_results = run_pearson(save_path + '/yhat_MAGIC_t_2.npy', target_file, roc, precision_recall_k, 'MAGIC t=2', i)
            individual_results.update(pearson_results)
            print(f"---> Running Pearson on MAGIC t=7 Data for DS{dataset} ")
            pearson_results = run_pearson(save_path + '/yhat_MAGIC_t_7.npy', target_file, roc, precision_recall_k, 'MAGIC t=7', i)
            individual_results.update(pearson_results)
            print(f"---> Running Pearson on MAGIC t=default Data for DS{dataset} ")
            pearson_results = run_pearson(save_path + '/yhat_MAGIC_t_auto.npy', target_file, roc, precision_recall_k, 'MAGIC t=default', i)
            individual_results.update(pearson_results)

    if genie:
        individual_results['DS'] = dataset
        individual_results['iter'] = iteration
        # get true regulator genes from SERGIO data
        reg_file = None
        if dataset == 1:
            reg_file = './SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt'
        elif dataset == 2:
            reg_file = './SERGIO/data_sets/De-noised_400G_9T_300cPerT_5_DS2/Regs_cID_5.txt'
        else:
            reg_file = 'SERGIO/data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Regs_cID_6.txt'
        master_regs = pd.read_table(reg_file, header=None, sep=',')
        master_regs = master_regs[0].values.astype(int).astype(str).tolist()

        regulators = []
        regulator_file = open(target_file, 'r')
        lines = regulator_file.readlines()
        for line in lines:
            row = line.split(',')
            num_regs_row = int(float(row[1]))
            if num_regs_row != 0:
                for i in range(2, num_regs_row + 2):
                    regulators.append(str(int(float(row[i]))))
        regs = list(set(regulators))
        regs = [i for i in regs if i not in master_regs]

        # Run GENIE3 on Clean Data
        print(f"---> Running GENIE3 on Clean Data for DS{dataset} ")
        gene_names = [str(i) for i in range(x.shape[1])]
        if not run_with_regs:
            regs = 'all'
            gene_names = None

        VIM_CLEAN = GENIE3(x, nthreads=12, ntrees=100, regulators=regs, gene_names=gene_names)        
        gt, rescaled_vim = gt_benchmark(VIM_CLEAN, target_file)
        if roc:
            roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
            #individual_results['DS' + str(i) + ' GENIE3 Clean ROC_AUC'] = float('%.2f'%(roc_score))
            individual_results['Clean'] = roc_score
        if precision_recall_k:
            k = range(1, gt.size)
            precision_k = precision_at_k(gt, rescaled_vim, k)
            #individual_results['DS' + str(i) + ' GENIE3 Clean Precision@k'] = precision_k

        # Run GENIE3 on Noisy Data
        print(f"---> Running GENIE3 on Noisy Data for DS{dataset} ")
        gene_names = [str(i) for i in range(y.shape[1])]
        VIM_NOISY = GENIE3(y, nthreads=12, ntrees=100, regulators=regs, gene_names=gene_names)       
        gt, rescaled_vim = gt_benchmark(VIM_NOISY, target_file)
        if roc:
            roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
            #individual_results['DS' + str(i) + ' GENIE3 Noisy ROC_AUC'] = float('%.2f'%(roc_score))
            individual_results['Noisy'] = roc_score
        if precision_recall_k:
            k = range(1, gt.size)
            precision_k = precision_at_k(gt, rescaled_vim, k)
            #individual_results['DS' + str(i) + ' GENIE3 Noisy Precision@k'] = precision_k

        # Run GENIE3 on SAUCIE Data
        if saucie:
            y_hat_saucie = np.load(save_path + '/yhat_SAUCIE' + file_extension +'.npy')

            print(f"---> Running GENIE3 on SAUCIE Data for DS{dataset} ")
            gene_names = [str(i) for i in range(y_hat_saucie.shape[1])]
            VIM_SAUCIE = GENIE3(y_hat_saucie, nthreads=12, ntrees=100, regulators=regs, gene_names=gene_names)
            gt, rescaled_vim = gt_benchmark(VIM_SAUCIE, target_file)
            # np.save(save_path + '/VIM_SAUCIE.npy', rescaled_vim)
            # np.save(save_path + '/gt_SAUCIE.npy', gt)
            if roc:
                roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                #individual_results['DS' + str(i) + ' GENIE3 SAUCIE ROC_AUC'] = float('%.2f'%(roc_score))
                individual_results['SAUCIE'] = roc_score
            if precision_recall_k:
                k = range(1, gt.size)
                precision_k = precision_at_k(gt, rescaled_vim, k)
                #individual_results['DS' + str(i) + ' GENIE3 SAUCIE Precision@k'] = precision_k
        
        # Run GENIE3 on scScope Data
        if scScope:
            y_hat_scscope = np.load(save_path + '/yhat_scScope' + file_extension +'.npy')
            y_hat_scScope  = y_hat_scscope.copy()
            y_hat_scScope[y_hat_scScope == 0] = 1e-5

            print(f"---> Running GENIE3 on scScope Data for DS{dataset} ")
            gene_names = [str(i) for i in range(y_hat_scscope.shape[1])]
            VIM_scScope = GENIE3(y_hat_scScope, nthreads=12, ntrees=100, regulators=regs, gene_names=gene_names)
            gt, rescaled_vim = gt_benchmark(VIM_scScope, target_file)
            if roc:
                roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                #individual_results['DS' + str(i) + ' GENIE3 scScope ROC_AUC'] = float('%.2f'%(roc_score))
                individual_results['scScope'] = roc_score
            if precision_recall_k:
                k = range(1, gt.size)
                precision_k = precision_at_k(gt, rescaled_vim, k)
                #individual_results['DS' + str(i) + ' GENIE3 scScope Precision@k'] = precision_k

        # Run GENIE3 on DeepImpute Data
        if deepImpute:
            y_hat_deepImpute = np.load(save_path + '/yhat_deepImpute' + file_extension +'.npy')

            print(f"---> Running GENIE3 on DeepImpute Data for DS{dataset} ")
            gene_names = [str(i) for i in range(y_hat_deepImpute.shape[1])]
            VIM_deepImpute = GENIE3(y_hat_deepImpute, nthreads=12, ntrees=100, regulators=regs, gene_names=gene_names)
            gt, rescaled_vim = gt_benchmark(VIM_deepImpute, target_file)
            np.save(save_path + '/VIM_deepImpute' + file_extension +'.npy', rescaled_vim)
            np.save(save_path + '/gt_deepImpute' + file_extension +'.npy', gt)
            print("saved deepimpute files")
            if roc:
                roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                #individual_results['DS' + str(i) + ' GENIE3 DeepImpute ROC_AUC'] = float('%.2f'%(roc_score))
                individual_results['DeepImpute'] = roc_score
            if precision_recall_k:
                k = range(1, gt.size)
                precision_k = precision_at_k(gt, rescaled_vim, k)
                #individual_results['DS' + str(i) + ' GENIE3 DeepImpute Precision@k'] = precision_k

        # Run GENIE3 on MAGIC Data
        if magic:
            #y_hat_magic_t2 = np.load(save_path + '/yhat_MAGIC_t_2.npy')
            y_hat_magic_t7 = np.load(save_path + '/yhat_MAGIC_t_7' + file_extension +'.npy')
            #y_hat_magic_t_auto = np.load(save_path + '/yhat_MAGIC_t_auto.npy')

            # print(f"---> Running GENIE3 on MAGIC t=2 for DS{dataset} ")
            # gene_names = [str(i) for i in range(y_hat_magic_t2.shape[1])]
            # VIM_MAGIC = GENIE3(y_hat_magic_t2, nthreads=12, ntrees=100, regulators='all', gene_names=gene_names)
            # gt, rescaled_vim = gt_benchmark(VIM_MAGIC, target_file)
            # if roc:
            #     roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
            #     individual_results['DS' + str(i) + ' GENIE3 MAGIC t=2 ROC_AUC'] = float('%.2f'%(roc_score))
            # if precision_recall_k:
            #     k = range(1, gt.size)
            #     precision_k = precision_at_k(gt, rescaled_vim, k)
            #     individual_results['DS' + str(i) + ' GENIE3 MAGIC t=2 Precision@k'] = precision_k

            print(f"---> Running GENIE3 on MAGIC t=7 for DS{dataset} ")
            gene_names = [str(i) for i in range(y_hat_magic_t7.shape[1])]
            VIM_MAGIC = GENIE3(y_hat_magic_t7, nthreads=12, ntrees=100, regulators=regs, gene_names=gene_names)
            gt, rescaled_vim = gt_benchmark(VIM_MAGIC, target_file)
            if roc:
                roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
                #individual_results['DS' + str(i) + ' GENIE3 MAGIC t=7 ROC_AUC'] = float('%.2f'%(roc_score))
                individual_results['MAGIC t=7'] = roc_score
            if precision_recall_k:
                k = range(1, gt.size)
                precision_k = precision_at_k(gt, rescaled_vim, k)
                #individual_results['DS' + str(i) + ' GENIE3 MAGIC t=7 Precision@k'] = precision_k
            
            # print(f"---> Running GENIE3 on MAGIC t=default for DS{dataset} ")
            # gene_names = [str(i) for i in range(y_hat_magic_t_auto.shape[1])]
            # VIM_MAGIC = GENIE3(y_hat_magic_t_auto, nthreads=12, ntrees=100, regulators='all', gene_names=gene_names)
            # gt, rescaled_vim = gt_benchmark(VIM_MAGIC, target_file)
            # if roc:
            #     roc_score = roc_auc_score(gt.flatten(), rescaled_vim.flatten())
            #     individual_results['DS' + str(i) + ' GENIE3 MAGIC t=default ROC_AUC'] = float('%.2f'%(roc_score))
            # if precision_recall_k:
            #     k = range(1, gt.size)
            #     precision_k = precision_at_k(gt, rescaled_vim, k)
            #     individual_results['DS' + str(i) + ' GENIE3 MAGIC t=default Precision@k'] = precision_k
        print(dataset, individual_results)
        return individual_results
    return #, count_methods

def create_correlation_plots(datasets):
    for i in tqdm(datasets):
        print(f"---> Calculating correlations for data from DS{dataset} ")
        ds_str = 'DS' + str(i)
        save_path = './zero_imputation_experiments/' + ds_str

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
        fig.suptitle(f'Correlation plots for DS{dataset} ')
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