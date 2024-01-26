import multiprocessing
from pathlib import Path
import shutil
from inferelator import inferelator_workflow, inferelator_verbose_level, MPControl
import pandas as pd
import typer
import csv

import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument('-expression_data', required=True, default=None, help="Simulated expression data")
parser.add_argument('-output_dir', required=True, default="./output/", help="Indicate the path for output.")
args = parser.parse_args()

# Function for obtaining the list of genes from expression file.
def get_gene_names_from_expression_file(expression_file):
    with open(expression_file, "r") as f:
        gene_list = [row.split(",")[0].replace('"', "") for row in f]
        gene_list = [str(g) for g in gene_list]
        if gene_list[0] == "":
            del gene_list[0]
    f.close()
    return gene_list

# Standarization of weights
def process_list(conf_list: pd.DataFrame):
    v_conf = conf_list["combined_confidences"]
    v_scaled = (v_conf - min(v_conf)) / (max(v_conf) - min(v_conf))
    conf_list["combined_confidences"] = v_scaled
    conf_list = conf_list.loc[conf_list["combined_confidences"] != 0]
    return conf_list

def inferelator(
    in_file,
    output_folder
):

    # Save gene names to file
    gene_names = get_gene_names_from_expression_file(in_file)
    gene_names_file = str(Path(in_file).parent) + "/gene_names.tsv"
    with open(gene_names_file, "w") as f:
        for gene in gene_names:
            f.write(gene + "\n")

    # Input file CSV to TSV format
    expression_file = str(Path(in_file).with_suffix(".tsv"))
    csv.writer(open(expression_file, 'w+'), delimiter='\t').writerows(csv.reader(open(in_file)))

    # Set verbosity level to "Normal"
    inferelator_verbose_level(-1)

    # Create a worker
    worker = inferelator_workflow(regression="bbsr", workflow="tfa")
    
    # Define the general run parameters
    worker.set_file_paths(input_dir=str(Path(in_file).parent),
                    expression_matrix_file=str(Path(expression_file).name),
                    tf_names_file=str(Path(gene_names_file).name),
                    output_dir="./to_remove")
    worker.set_network_data_flags(use_no_gold_standard=True,
                    use_no_prior=True)
    worker.set_file_properties(expression_matrix_columns_are_genes=False)
    worker.set_run_parameters(num_bootstraps=50, random_seed=100)

    # Run
    net = worker.run()

    # Remove temp files
    Path(expression_file).unlink()
    Path(gene_names_file).unlink()

    # Remove predefined output files
    shutil.rmtree(Path("./to_remove"))

    # Get confidence list
    conf_list = net.network.iloc[:, 0:3]

    # Standardization
    conf_list = process_list(conf_list)

    # Save list
    conf_list.to_csv(
        f"./{output_folder}/GRN_INFERELATOR.csv", header=False, index=False
    )


# Multiprocessing needs to be protected with the if __name__ == 'main' pragma
if __name__ == '__main__':
    MPControl.set_multiprocess_engine("multiprocessing")
    MPControl.client.processes = multiprocessing.cpu_count()
    MPControl.connect()
    inferelator(in_file=args.expression_data, output_folder=args.output_dir)
    #typer.run(inferelator)
