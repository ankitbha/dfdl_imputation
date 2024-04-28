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

def add_edge_method(regulators, targets, master_regs, n_genes):
    chosen_target = random.randint(0, n_genes - 1)
    regs = list(regulators.keys())
    chosen_regulator = random.choice(regs)
    while chosen_target in [t[0] for t in regulators[chosen_regulator]] or chosen_target in master_regs or chosen_target in regs or chosen_target == chosen_regulator:
        chosen_target = random.randint(0, n_genes - 1)
    while chosen_regulator in [t[0] for t in targets[chosen_target]] or chosen_target == chosen_regulator:
        chosen_regulator = random.choice(regs)
    return chosen_regulator, chosen_target

def run_dfs(node, graph, visited, rec_stack):
    visited[node] = True
    rec_stack[node] = True

    for neighbour in graph.get(node, []):
        if neighbour not in visited:
            visited[neighbour] = False
            rec_stack[neighbour] = False
        if not visited[neighbour]:
            if run_dfs(neighbour, graph, visited, rec_stack):
                return True
        elif rec_stack[neighbour]:
            return True
    rec_stack[node] = False
    return False

def add_edge_and_check_cycle(chosen_regulator, chosen_target, graph):
    if chosen_regulator not in graph:
        graph[chosen_regulator] = []
    graph[chosen_regulator].append(chosen_target)

    all_nodes = set(graph.keys()) | {node for neighbours in graph.values() for node in neighbours}

    visited = {node: False for node in all_nodes}
    rec_stack = {node: False for node in all_nodes}

    for node in graph:
        if not visited[node]:
            if run_dfs(node, graph, visited, rec_stack):
                return True
    return False

from SERGIO.SERGIO.sergio import sergio
def modified_sergio(input_file, reg_file, ind, n_genes=1200, n_bins=9, n_sc=300, file_extension = ''):
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
    save_path = './imputations/' + ds_str
    
    # Save simulated data variants
    np.save(save_path + '/DS6_clean' + file_extension, expr_clean )
    np.save(save_path + '/DS6_expr' + file_extension, expr)
    cmat_clean = sim.convert_to_UMIcounts(expr)
    cmat_clean = np.concatenate(cmat_clean, axis = 1)
    np.save(save_path + '/DS6_clean_counts' + file_extension, cmat_clean)

    # Add Technical Noise - Steady State Simulations
    expr_O = sim.outlier_effect(expr, outlier_prob = 0.01, mean = 5, scale = 1)
    return expr, expr_clean

    # To-implement
    # libFactor, expr_O_L = sim.lib_size_effect(expr_O, mean = 4.5, scale = 0.7)
    # binary_ind = sim.dropout_indicator(expr_O_L, shape = 8, percentile = 45)
    # expr_O_L_D = np.multiply(binary_ind, expr_O_L)
    # count_matrix = sim.convert_to_UMIcounts(expr_O_L_D)
    # count_matrix = np.concatenate(count_matrix, axis = 1)
    # np.save(save_path + '/DS6_noisy' + file_extension, count_matrix)

def new_mean_process_iteration(iteration, target_file, regs_path, master_regs, load_dir, add_edge, multiple_edges, imp_dir, dataset_id, file_extension='', clean='clean', normalize=False):
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

    chosen_pairs = []
    chosen_tuples = []
    target_sub_map = {}
    for target in targets:
        target_sub_map[target] = 0

    regulator_adjacency_list = {}
    for reg in regulators:
        for tuple in regulators[reg]:
            target = tuple[0]
            if reg not in regulator_adjacency_list.keys():
                regulator_adjacency_list[reg] = [target]
            else:
                regulator_adjacency_list[reg].append(target) 
    
    if multiple_edges:
        num_edges = [len(targs) for targs in regulators.values()]
        total_mods = int(sum(num_edges) * 0.3)
        for i in range(total_mods):
            add = random.choice([True, False])
            if add:
                chosen_regulator, chosen_target = add_edge_method(regulators, targets, master_regs, genes)
                # has_cycle = True
                # cycle_iter = 50
                # while has_cycle:
                #     chosen_regulator, chosen_target = add_edge_method(regulators, targets, master_regs, genes)
                #     has_cycle = add_edge_and_check_cycle(chosen_regulator, chosen_target, regulator_adjacency_list)
                #     cycle_iter -= 1
                # if cycle_iter == 0:
                #     continue
                #target_sub_map[chosen_target] -= 1
            else:
                chosen_target = random.choice(list(targets.keys()))
                while all([t[0] in master_regs for t in targets[chosen_target]]) or len(targets[chosen_target]) - target_sub_map[chosen_target] <= 1 or chosen_target in master_regs:
                    chosen_target = random.choice(list(targets.keys()))
                target_sub_map[chosen_target] += 1
                chosen_regulator = random.choice([t[0] for t in targets[chosen_target] if t[0] not in master_regs])
                while chosen_regulator in master_regs or chosen_regulator == chosen_target:
                    chosen_regulator = random.choice([t[0] for t in targets[chosen_target] if t[0] not in master_regs])
            temp_tuple = (chosen_regulator, chosen_target)
            if temp_tuple not in chosen_pairs:
                chosen_pairs.append(temp_tuple)
                chosen_tuples.append((chosen_regulator, chosen_target, add))
            else:
                i -= 1
    else:
        add = add_edge
        if add:
            #has_cycle = True
            #while has_cycle:
            chosen_regulator, chosen_target = add_edge_method(regulators, targets, master_regs, genes)
            #has_cycle = add_edge_and_check_cycle(chosen_regulator, chosen_target, regulator_adjacency_list)
        else:
            chosen_target = random.choice(list(targets.keys()))
            while all([t[0] in master_regs for t in targets[chosen_target]]) or len(targets[chosen_target]) <= 1:
                chosen_target = random.choice(list(targets.keys()))
            chosen_regulator = random.choice([t[0] for t in targets[chosen_target] if t[0] not in master_regs])
            while chosen_regulator in master_regs:
                chosen_regulator = random.choice([t[0] for t in targets[chosen_target] if t[0] not in master_regs])
        chosen_tuples.append((chosen_regulator, chosen_target, add))
    
    chosen_regulators = [pair[0] for pair in chosen_tuples]
    chosen_targets = [pair[1] for pair in chosen_tuples]
    add_subtract = [pair[2] for pair in chosen_tuples]
    
    new_edges = []
    for i in range(len(chosen_targets)):
        chosen_target = chosen_targets[i]
        chosen_regulator = chosen_regulators[i]
        add = add_subtract[i]
        if add:
            # if multiple_edges:
            # has_cycle = add_edge_and_check_cycle(chosen_regulator, chosen_target, regulator_adjacency_list)
            # cycle_iter = 50
            # while has_cycle:
            #     chosen_regulator, chosen_target = add_edge_method(regulators, targets, master_regs, genes)
            #     has_cycle = add_edge_and_check_cycle(chosen_regulator, chosen_target, regulator_adjacency_list)
            #     cycle_iter -= 1
            #     if cycle_iter == 0:
            #         break
            # if cycle_iter == 0:
            #     continue
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
            if chosen_regulator not in regulators:
                regulator_adjacency_list[chosen_regulator] = [chosen_target]
            else:
                regulator_adjacency_list[chosen_regulator].append(chosen_target)

            new_edges.append((chosen_regulator, chosen_target, random_hill, 2.0))
        else:
            if chosen_target in targets:
                targets[chosen_target] = [t for t in targets[chosen_target] if t[0] != chosen_regulator]
            if chosen_regulator in regulators:
                regulators[chosen_regulator] = [r for r in regulators[chosen_regulator] if r[0] != chosen_target]

    with open(temp_target, 'w') as file_copy:
        for ind, target in enumerate(targets.items()):
            t = float(target[0])
            len_regs = float(len(target[1]))
            regs = [str(x[0]) for x in target[1]]
            hill_values = [str(x[1]) for x in target[1]]
            ns = [str(x[2]) for x in target[1]]
            file_copy.write(f'{t},{len_regs},{",".join(regs)},{",".join(hill_values)},{",".join(ns)}\n')

    #expr_data, clean_data = modified_sergio(temp_target, regs_path, dataset_id, file_extension=file_extension)
    #return expr_data, clean_data, chosen_tuples
    run_sergio(temp_target, regs_path, dataset_id, file_extension=file_extension)
    sergio_df = pd.DataFrame(np.load(os.path.join(load_dir, f"DS6_{clean}.npy")))
    if normalize:
        other_df = pd.DataFrame(np.load(os.path.join(load_dir, f"DS6_{clean}{file_extension}.npy")))
        total_counts = other_df.sum(axis=0)
        normalized_df = other_df.divide(total_counts, axis='columns') #* 1e6 # counts per million?
        np.save(os.path.join(load_dir, f"DS6_{clean}{file_extension}.npy"), normalized_df.to_numpy())
    other_df = pd.DataFrame(np.load(os.path.join(load_dir, f"DS6_{clean}{file_extension}.npy")))
    
    differences = other_df - sergio_df
    gaussian_params = differences.apply(fit_gaussian, axis=1)
    gaussian_params_with_index = gaussian_params.reset_index().rename(columns={'index': 'original_index'})
    
    final_ranks = []
    for iter, target in enumerate(chosen_targets):
        chosen_reg = chosen_regulators[iter]
        add_sub = add_subtract[iter]
        ranked_gaussian_params = gaussian_params_with_index.sort_values(by='mean', ascending=False)#ascending=(not add_sub))
        rank = ranked_gaussian_params.index.get_loc(target)
        rank += 1
        value = ranked_gaussian_params.loc[ranked_gaussian_params['original_index'] == target]
        final_ranks.append((chosen_reg, target, add_sub, rank, value))

    return ranked_gaussian_params, final_ranks, temp_target, file_extension, iteration