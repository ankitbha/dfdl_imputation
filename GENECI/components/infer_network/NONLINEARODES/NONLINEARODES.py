import math
import time

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBRegressor
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument('-expression_data', required=True, default=None, help="Simulated expression data")
parser.add_argument('-output_dir', required=True, default="./output/", help="Indicate the path for output.")
args = parser.parse_args()

def get_importances(expr_data, gene_names, regulators, param={}):

    time_start = time.time()

    ngenes = expr_data.shape[1]

    # Get the indices of the candidate regulators
    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]

    # Learn an ensemble of trees for each target gene, and compute scores for candidate regulators
    VIM = np.zeros((ngenes, ngenes))

    for i in tqdm(range(ngenes)):
        input_idx = idx.copy()
        if i in input_idx:
            input_idx.remove(i)
        vi = get_importances_single(expr_data, i, input_idx, param)
        VIM[i, :] = vi

    time_end = time.time()
    print("Elapsed time: %.2f seconds" % (time_end - time_start))

    return VIM


def get_importances_single(expr_data, output_idx, input_idx, param):

    ngenes = expr_data.shape[1]

    # Expression of target gene
    output = expr_data.iloc[:, output_idx]

    # Normalize output data
    output = output / np.std(output)

    expr_data_input = expr_data.iloc[:, input_idx]
    treeEstimator = XGBRegressor(**param)

    # Learn ensemble of trees
    treeEstimator.fit(expr_data_input, output)

    # Compute importance scores
    feature_importances = treeEstimator.feature_importances_
    vi = np.zeros(ngenes)
    vi[input_idx] = feature_importances

    return vi


def get_scores(VIM, gold_edges, gene_names, regulators):

    idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    pred_edges = [
        (gene_names[j], gene_names[i], score)
        for (i, j), score in np.ndenumerate(VIM)
        if i != j and j in idx
    ]
    pred_edges = pd.DataFrame(pred_edges)
    pred_edges.sort_values(2, ascending=False, inplace=True)
    # Take the top 100000 predicted results
    pred_edges = pred_edges.iloc[:100000]
    final = pd.merge(pred_edges, gold_edges, on=[0, 1], how="inner")
    auroc = roc_auc_score(final["2_y"], final["2_x"])
    aupr = average_precision_score(final["2_y"], final["2_x"])

    return auroc, aupr


def process_list(conf_list: pd.DataFrame):
    v_conf = conf_list["Confidence"]
    v_scaled = (v_conf - min(v_conf)) / (max(v_conf) - min(v_conf))
    conf_list["Confidence"] = v_scaled
    conf_list = conf_list.loc[conf_list["Confidence"] != 0]
    conf_list = conf_list.sort_values(by="Confidence", ascending=False)
    return conf_list


def non_linear_odes(
    in_file: str = typer.Argument(..., help="CSV input file"),
    output_folder: str = typer.Argument(..., help="Path to output folder"),
):

    # Load the expression matrix
    print("---> working ... reading expression matrix")
    ex_matrix = pd.read_csv(in_file, index_col=0).T

    # Define gene identifiers
    print("---> working ... defining gene identifiers")
    gene_names = list(ex_matrix.columns.values)

    # Consider all genes as regulators (since no other information is available)
    regulators = gene_names.copy()

    # Set xgboost parameters
    xgb_param = dict(
        n_jobs=-1,
        max_depth=5,
        importance_type="weight",
        n_estimators=round(len(gene_names) * 10 / math.log10(len(gene_names))),
    )

    # Infer gene regulatory network
    print("---> working ... inferring gene regulatory network")
    conf_matrix = get_importances(
        ex_matrix, gene_names=gene_names, regulators=regulators, param=xgb_param
    )

    # Get confidence list from matrix
    print("---> working ... getting confidence list from matrix")
    conf_matrix = pd.DataFrame(conf_matrix, columns=gene_names, index=gene_names)
    np.fill_diagonal(conf_matrix.values, np.nan)
    conf_list = conf_matrix.stack().reset_index()
    conf_list.columns = ["Source", "Target", "Confidence"]

    # Standardization
    print("---> working ... standardizing confidence list")
    conf_list = process_list(conf_list)

    # Save list
    print("---> working ... saving confidence list")
    conf_list.to_csv(
        f"./{output_folder}/GRN_NONLINEARODES.csv", header=False, index=False
    )


if __name__ == "__main__":
    non_linear_odes(in_file=args.expression_data, output_folder=args.output_dir)
    #typer.run(non_linear_odes)
