from consolidated_runs import run_sergio
from Pearson.pearson import Pearson
from scipy.stats import norm
import numpy as np
import pandas as pd
import os
import random

def fit_gaussian(row):
    mean, std = norm.fit(row)
    return pd.Series({'mean': mean, 'std': std})

def process_iteration(iteration, target_file, regs_path, master_regs, load_dir, imp_dir, i, file_extension=''):
    regulators = {}
    targets = {}
    chosen_pair = None
    temp_target = None
    with open(target_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            values = line.split(',')
            target = float(values[0])
            num_regs = int(float(values[1]))
            #print(values, target, num_regs)
            regs = [float(v) for v in values[2:2 + num_regs]]
            hill_values = [float(v) for v in values[2 + num_regs:]]
            other_hill_values = [v for v in hill_values[int(len(hill_values) / 2):]]
            for r_index, zipped in enumerate(zip(regs, hill_values, other_hill_values)):
                reg = zipped[0]
                hill = zipped[1]
                other_hill = zipped[2]
                if reg not in regulators:
                    regulators[reg] = [(target, hill, other_hill)]
                else:
                    regulators[reg].append((target, hill, other_hill))
                if target not in targets:
                    targets[target] = [(reg, hill, other_hill)]
                else:
                    targets[target].append((reg, hill, other_hill))
    file.close()
    temp_target = target_file.replace('.txt', f'_iter{iteration}_temp.txt')
    # Get number of genes to choose a target
    imp_dir = os.path.join(os.getcwd(), 'imputations')
    load_dir = os.path.join(imp_dir, f'DS{i}')
    expr_name = 'DS6_expr.npy'
    expr = np.load(os.path.join(load_dir, expr_name))
    genes = expr.shape[1]
    #print(genes)
    chosen_target = random.randint(0, genes - 1)

    chosen_regulator = random.choice(list(regulators.keys()))
    

    while chosen_target in [t[0] for t in regulators[chosen_regulator]] or chosen_target in master_regs or chosen_target == chosen_regulator:
        chosen_target = random.randint(0, genes - 1)
    
    min_hill = 1.0
    max_hill = 3.0
    if chosen_target in targets:
        min_hill = np.min([float(hill[1]) for hill in targets[chosen_target]])
        max_hill = np.max([float(hill[1]) for hill in targets[chosen_target]])
    random_hill = random.uniform(min_hill, max_hill)

    if chosen_target not in targets:
        targets[chosen_target] = [(chosen_regulator, random_hill, 2.0)]
    else:
        targets[chosen_target].append((chosen_regulator, random_hill, 2.0))
    
    regulators[chosen_regulator].append((chosen_target, random_hill, 2.0))
    chosen_pair = (chosen_regulator, chosen_target)
    #print(chosen_pair)
    with open(temp_target, 'w') as file_copy:
        for ind, target in enumerate(targets.items()):
            t = float(target[0])
            regs = float(len(target[1]))
            regulators = [str(x[0]) for x in target[1]]
            hill_values = [str(x[1]) for x in target[1]]
            other_hill_values = [str(x[2]) for x in target[1]]
            file_copy.write(f'{t},{regs},{",".join(regulators)},{",".join(hill_values)},{",".join(other_hill_values)}\n')
    
    #print(chosen_pair, min_hill, max_hill, random_hill)
    run_sergio(temp_target, regs_path, i, file_extension=file_extension)

    clean_df = pd.DataFrame(np.load(os.path.join(load_dir, f"DS6_clean{file_extension}.npy")))
    pearson = Pearson(np.transpose(clean_df), '')
    p_values = pearson.values
    np.fill_diagonal(p_values, 0)
    pearson = pd.DataFrame(p_values, index=pearson.columns, columns=pearson.columns)
    return pearson, chosen_pair, temp_target, file_extension

def new_mean_process_iteration(iteration, target_file, regs_path, master_regs, load_dir, add_edge, imp_dir, dataset_id, file_extension=''):
    regulators = {}
    targets = {}
    chosen_pair = None
    temp_target = None
    with open(target_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            values = line.split(',')
            target = float(values[0])
            num_regs = int(float(values[1]))
            #print(values, target, num_regs)
            regs = [float(v) for v in values[2:2 + num_regs]]
            hill_values = [float(v) for v in values[2 + num_regs:]]
            ns = [v for v in hill_values[int(len(hill_values) / 2):]]
            for r_index, zipped in enumerate(zip(regs, hill_values, ns)):
                reg = zipped[0]
                hill = zipped[1]
                other_hill = zipped[2]
                if reg not in regulators:
                    regulators[reg] = [(target, hill, other_hill)]
                else:
                    regulators[reg].append((target, hill, other_hill))
                if target not in targets:
                    targets[target] = [(reg, hill, other_hill)]
                else:
                    targets[target].append((reg, hill, other_hill))
    
    temp_target = target_file.replace('.txt', f'_iter{iteration}_temp.txt')
    expr_name = 'DS6_expr.npy'
    expr = np.load(os.path.join(load_dir, expr_name))
    genes = expr.shape[1]
    chosen_target = random.randint(0, genes - 1)
    chosen_regulator = random.choice(list(regulators.keys()))
    while chosen_target in [t[0] for t in regulators[chosen_regulator]] or chosen_target in master_regs or chosen_target == chosen_regulator:
        chosen_target = random.randint(0, genes - 1)

    if add_edge:
        min_hill = 1.0
        max_hill = 3.0
        if chosen_target in targets:
            min_hill = np.min([float(hill[1]) for hill in targets[chosen_target]])
            max_hill = np.max([float(hill[1]) for hill in targets[chosen_target]])
        random_hill = random.uniform(min_hill, max_hill)

    clean_df = pd.DataFrame(np.load(os.path.join(load_dir, f"DS6_clean.npy")))
    if chosen_target not in targets:
            targets[chosen_target] = [(chosen_regulator, random_hill, 2.0)]
    else:
        if add_edge:
            targets[chosen_target].append((chosen_regulator, random_hill, 2.0))
        else:
            pass # need to remove
    
    if add_edge:
        regulators[chosen_regulator].append((chosen_target, random_hill, 2.0))
    else:
        pass # need to remove
    chosen_pair = (chosen_regulator, chosen_target)

    with open(temp_target, 'w') as file_copy:
        for ind, target in enumerate(targets.items()):
            t = float(target[0])
            len_regs = float(len(target[1]))
            regs = [str(x[0]) for x in target[1]]
            hill_values = [str(x[1]) for x in target[1]]
            ns = [str(x[2]) for x in target[1]]
            file_copy.write(f'{t},{len_regs},{",".join(regs)},{",".join(hill_values)},{",".join(ns)}\n')

    run_sergio(temp_target, regs_path, dataset_id, file_extension=file_extension)
    other_df = pd.DataFrame(np.load(os.path.join(load_dir, f"DS6_clean{file_extension}.npy")))
    
    differences = other_df - clean_df
    gaussian_params = differences.apply(fit_gaussian, axis=1)
    gaussian_params_with_index = gaussian_params.reset_index().rename(columns={'index': 'original_index'})
    ranked_gaussian_params = gaussian_params_with_index.sort_values(by='mean', ascending=False)
    rank = ranked_gaussian_params.index.get_loc(chosen_target)
    rank += 1

    return ranked_gaussian_params, chosen_pair, rank, temp_target, file_extension