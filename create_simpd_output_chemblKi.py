''' process the results of a SIMPD run on ChEMBL data and create
a set of csv files for use in other projects along with a data
description YAML for use with intake (https://github.com/intake/intake)

Author: Greg Landrum (glandrum@ethz.ch)
'''
import intake
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
# the RDKit standardizer is noisy, disable the logger:
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.info')
import numpy as np
import ga_lib_3 as ga_lib
import sys
import glob
import math
import os.path
import pandas as pd

favorFG = True

# -----------------------
# reading the molecules into a dataframe
# The rest of the setup code assumes that the data frame df has a column
# named 'mol' with rdkit molecules. If you don't have that, you could just use a list
# of molecules and adjust the code below to reflect that
bioactivities = intake.open_catalog('./datasets/targets.yaml')
assay_accum = []

for target in bioactivities:
    ds = bioactivities[target]

    print('------------\nLoading data')
    df = ds.read()

    # randomize the order of the molecules:
    # this is easy with a data frame, but you could also random.shuffle() a list if the mols are in a list.
    df = df.sample(frac=1, random_state=0xf00d)

    # generate the binned activity values
    pActs = np.array(df.pchembl_value.to_list())
    upper, lower = ga_lib.get_imbalanced_bins(pActs,
                                              tgt_frac=0.4,
                                              active_inactive_offset=0.0)
    df = df[(pActs >= upper) | (pActs <= lower)]
    binned = [1 if x >= upper else 0 for x in df.pchembl_value]
    df['active'] = binned

    mols = [
        rdMolStandardize.ChargeParent(Chem.MolFromSmiles(tmp_smi))
        for tmp_smi in df['canonical_smiles'].to_numpy(dtype=str)
    ]
    df['canonical_smiles'] = [Chem.MolToSmiles(x) for x in mols]

    import pickle
    import gzip

    loc = 'nogap_alt_results_balanced'
    fn = f'./{loc}/{target}_CLUSTERS_SPLIT.500.300.altscenario_0.pkl.gz'
    if not os.path.exists(fn):
        print(f'{fn} not found, skipping')
        continue
    out_loc = 'chemblKi_SIMPD'
    with gzip.open(fn, 'rb') as inf:
        train_inds, test_inds = pickle.load(inf)
        Fs, Gs, Xs = pickle.load(inf)
    weights = [1] * Fs.shape[1]
    if favorFG:
        weights[-2] = 10
        weights[-1] = 5
    scores = ga_lib.score_pareto_solutions(Fs, weights)
    bscores = np.argsort(scores)
    best = bscores[0]
    train_mask = ~Xs[best]
    test_mask = Xs[best]
    df['split'] = ['train'] * len(df)
    df.loc[test_mask, 'split'] = 'test'
    outn = f'./datasets/{out_loc}/{target}.csv'
    df = df.drop(columns=['mol'])
    df.to_csv(outn, index=False)
    md = ds.describe()
    yaml = f'''  {target}:
    description: "{md['description']}"
    notes: |
      All compounds have been salt stripped and neutralized using the RDKit's ChargeParent standardizer.
      The ChEMBL IDs in this file correspond to the original structure : the one which was measured

    args:
      filename: "{{{{ CATALOG_DIR }}}}/{out_loc}/{target}.csv"
      smilesColumn: canonical_smiles
    driver: intake_rdkit.smiles.SmilesSource
    metadata:
'''
    for k, v in md['metadata'].items():
        yaml += f'      {k}: {v}\n'
    yaml += f'      is_log_data: true\n'
    yaml += f'      activity bin: {upper}\n'
    yaml += f'      Compounds: {len(df)}\n'
    yaml += f'      Num Active: {sum(binned)}\n'
    yaml += f'      Num Inactive: {len(df) - sum(binned)}\n'
    assay_accum.append(yaml)

hdr = f'''metadata:
  version: 1
  creator:
    name: greg landrum
    email: glandrum@ethz.ch
  summary: |
    Collection of datasets with pchembl_values for bioactivity prediction.

    Each row includes the reported value. Only values without data_validity_comments are included
    Active/inactive class assignments were done to give a 40/60 inactive/active ratio
    The suggested train/test split was created using the SIMPD algorithm
    All compounds have been salt stripped and neutralized using the RDKit's ChargeParent standardizer
  split_column: split
  class_column: active
  activity_column: standard_value

sources:
'''
with open('./datasets/chemblKi_SIMPD.yaml', 'w+') as outf:
    print(hdr + '\n'.join(assay_accum), file=outf)
