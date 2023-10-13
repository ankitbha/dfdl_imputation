import numpy as np
import pandas as pd
from SERGIO.sergio import sergio

sim = sergio(number_genes=1200, number_bins = 9, number_sc = 300, noise_params = 1, decays=0.8, sampling_state=15, noise_type='dpd')
sim.build_graph(input_file_taregts ='data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Interaction_cID_6.txt',\
                        input_file_regs='data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Regs_cID_6.txt', shared_coop_state=2)
sim.simulate()
expr = sim.getExpressions()
expr_clean = np.concatenate(expr, axis = 1)

np.save('imputation_data/DS6_clean', expr_clean)
np.save('imputation_data/DS6_expr', expr)

cmat_clean = sim.convert_to_UMIcounts(expr)
cmat_clean = np.concatenate(cmat_clean, axis = 1)
np.save('imputation_data/DS6_clean_counts', cmat_clean)

"""
Add outlier genes
"""
expr_O = sim.outlier_effect(expr, outlier_prob = 0.01, mean = 5, scale = 1)

"""
Add Library Size Effect
"""
libFactor, expr_O_L = sim.lib_size_effect(expr_O, mean = 4.5, scale = 0.7)

"""
Add Dropouts
"""
binary_ind = sim.dropout_indicator(expr_O_L, shape = 8, percentile = 45)
expr_O_L_D = np.multiply(binary_ind, expr_O_L)

"""
Convert to UMI count
"""
count_matrix = sim.convert_to_UMIcounts(expr_O_L_D)

"""
Make a 2d gene expression matrix
"""
count_matrix = np.concatenate(count_matrix, axis = 1)

np.save('imputation_data/DS6_45', count_matrix)
