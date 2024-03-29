import intake
from rdkit import DataStructs
import numpy as np
import ga_lib_3 as ga_lib
import sys
import math

import argparse

train_frac_active = -1

parser = argparse.ArgumentParser()
parser.add_argument("target")
parser.add_argument("--strategy",
                    "-s",
                    choices=['DIVERSE', 'CLUSTERS_SPLIT'],
                    default='CLUSTERS_SPLIT')
parser.add_argument("--popSize", "-p", type=int, default=500)
parser.add_argument("--nGens", "-g", type=int, default=300)
parser.add_argument("--useAlternateSpatialStatsObjectives",
                    "-a",
                    default=True,
                    action='store_false')
parser.add_argument("--numThreads", type=int, default=1)
parser.add_argument("--balanced", "-b", default=False, action="store_true")
args = parser.parse_args()

# Name of the dataset to use:
target = args.target
pop_size = args.popSize
ngens = args.nGens
strategy = args.strategy
altStats = args.useAlternateSpatialStatsObjectives
isBalanced = args.balanced

if not altStats:
    raise NotImplementedError("not done")

# -----------------------
# reading the molecules into a dataframe
# The rest of the setup code assumes that the data frame df has a column
# named 'mol' with rdkit molecules. If you don't have that, you could just use a list
# of molecules and adjust the code below to reflect that
bioactivities = intake.open_catalog('./datasets/targets.yaml')

ds = bioactivities[target]

print(f'------------  Target {target}\nLoading data')
df = ds.read()

# randomize the order of the molecules:
# this is easy with a data frame, but you could also random.shuffle() a list if the mols are in a list.
df = df.sample(frac=1, random_state=0xf00d)

# generate the binned activity values
pActs = np.array(df.pchembl_value.to_list())
if not isBalanced:
    upper, lower = ga_lib.get_imbalanced_bins(pActs,
                                              active_inactive_offset=0.0)
else:
    upper, lower = ga_lib.get_imbalanced_bins(pActs,
                                              tgt_frac=0.4,
                                              active_inactive_offset=0.0)
df = df[(pActs >= upper) | (pActs <= lower)]
binned = [1 if x >= upper else 0 for x in pActs]
df['active'] = ['active' if x == 1 else 'inactive' for x in binned]

frac_active = np.sum(binned) / len(binned)

for j, dfrac in enumerate(ga_lib.delta_test_active_frac_vals):
    if not isBalanced:
        outfn = './nogap_alt_results_imbalanced'
    else:
        outfn = './nogap_alt_results_balanced'

    outfn += f'/{target}_{strategy}.{pop_size}.{ngens}.altscenario_{j}.pkl.gz'
    print(f'------------\nRunning GA, iter {dfrac:.2f}')
    train_inds, tests_inds, res = ga_lib.run_GA_SIMPD(
        df,
        strategy,
        pop_size,
        ngens,
        verbose=True,
        numThreads=args.numThreads,
        return_random_result=False,
        smilesCol='canonical_smiles',
        actCol='active',
        targetTrainFracActive=-1,
        targetTestFracActive=-1,
        targetDeltaTestFracActive=dfrac)

    print(f"{len(train_inds)} solutions")

    import pickle
    import gzip
    with gzip.open(outfn, 'wb+') as outf:
        pickle.dump((train_inds, tests_inds), outf)
        pickle.dump((res.F, res.G, res.X), outf)
