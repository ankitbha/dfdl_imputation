import argparse
import sys, os
import gzip
import pandas as pd
import numpy as np


def Pearson(file, outfile):
    counts = pd.DataFrame(file)
    pearson = counts.corr('pearson').abs()
    pearson.index = pearson.index.rename('Gene')

    pearson = pearson.reset_index(drop=True)
    pearson = pearson.rename(columns={'Gene': 'regulator'})
    #np.fill_diagonal(pearson.values, 0)
    return pearson
    #  pearson.to_csv(args.fout, sep=',', index=True, header=True, compression='gzip')

    interactions = pd.melt(pearson, id_vars=['regulator'],
                           var_name='target', value_name='score')
    interactions['score'] = interactions['score'].abs()
    interactions = interactions.loc[interactions.regulator != interactions.target]

    # Output
    compression = 'gzip' if args.fout.endswith('gz') else None
    interactions.to_csv(args.fout, sep='\t', index=False, header=True, compression=compression)
