''' process the results of a SIMPD run on ChEMBL data and create
a set of csv files for use in other projects along with a data
description YAML for use with intake (https://github.com/intake/intake)

Author: Greg Landrum (glandrum@ethz.ch)
'''
import intake
from rdkit import Chem
import numpy as np
import ga_lib_3 as ga_lib
import sys
import glob
import math
import os.path
import pandas as pd




assays = ['CHEMBL1267247', 'CHEMBL3705464', 'CHEMBL3705282', 'CHEMBL1267250', 'CHEMBL3705791', 'CHEMBL3705655', 'CHEMBL3705542', 'CHEMBL3705813', 'CHEMBL3705362', 'CHEMBL3705790', 'CHEMBL1267248', 'CHEMBL3705647', 'CHEMBL3705924', 'CHEMBL3705899', 'CHEMBL3706310', 'CHEMBL3705960', 'CHEMBL1267245', 'CHEMBL3705971', 'CHEMBL3706037', 'CHEMBL3880337', 'CHEMBL3706089', 'CHEMBL3706316', 'CHEMBL3887061', 'CHEMBL3734252', 'CHEMBL3887063', 'CHEMBL3887188', 'CHEMBL3880340', 'CHEMBL3880338', 'CHEMBL3887296', 'CHEMBL3887679', 'CHEMBL3887033', 'CHEMBL3734552', 'CHEMBL3707951', 'CHEMBL3707962', 'CHEMBL3888268', 'CHEMBL3721139', 'CHEMBL3887987', 'CHEMBL3706373', 'CHEMBL3887758', 'CHEMBL3888087', 'CHEMBL3887945', 'CHEMBL3887759', 'CHEMBL3888190', 'CHEMBL3887757', 'CHEMBL3888194', 'CHEMBL3887796', 'CHEMBL3887849', 'CHEMBL3888295', 'CHEMBL3887887', 'CHEMBL3888977', 'CHEMBL3889139', 'CHEMBL3888980', 'CHEMBL3889082', 'CHEMBL3889083', 'CHEMBL3888825', 'CHEMBL3888966'] 

favorFG = True

# -----------------------
# reading the molecules into a dataframe
# The rest of the setup code assumes that the data frame df has a column
# named 'mol' with rdkit molecules. If you don't have that, you could just use a list
# of molecules and adjust the code below to reflect that
catalog = intake.open_catalog('./datasets/public_data.yaml')
bioactivities = catalog.assays

assay_accum = []
for assay in assays:
    ds = bioactivities[assay]

    print('------------\nLoading data')
    df = ds.read()

    # randomize the order of the molecules:
    # this is easy with a data frame, but you could also random.shuffle() a list if the mols are in a list.
    df = df.sample(frac=1, random_state=0xf00d)

    # generate the binned activity values
    pActs = np.array(
        [-1 * math.log10(x*1e-9) if x > 0 else 0 for x in df.standard_value])
    df['pActivity'] = pActs
    upper, lower = ga_lib.get_imbalanced_bins(pActs,tgt_frac=0.4,active_inactive_offset=0.0)
    df = df[(pActs >= upper) | (pActs <= lower)]
    binned = [1 if x >= upper else 0 for x in df.pActivity]
    df['active'] = binned

    import pickle
    import gzip

    loc = 'nogap_alt_results_balanced_assay'
    fn = f'./{loc}/{assay}_CLUSTERS_SPLIT.500.300.altscenario_0.pkl.gz'
    if not os.path.exists(fn):
        print(f'{fn} not found, skipping')
        continue
    out_loc = 'chemblassay_SIMPD'
    with gzip.open(fn,'rb') as inf:
        train_inds,test_inds = pickle.load(inf)
        Fs,Gs,Xs = pickle.load(inf)
    weights = [1]*Fs.shape[1]
    if favorFG:
        weights[-2] = 10
        weights[-1] = 5
    scores = ga_lib.score_pareto_solutions(Fs,weights)
    bscores = np.argsort(scores)
    best = bscores[0]
    train_mask = ~Xs[best]
    test_mask = Xs[best]
    df['split'] = ['train']*len(df)
    df['split'][test_mask] = 'test'
    outn = f'./{out_loc}/{assay}.csv'
    df = df.drop(columns=['mol'])
    df.to_csv(outn)
    md = ds.describe()
    yaml = f'''  {assay}:
    description: "{md['description']}"
    args:
      filename: "{{{{ CATALOG_DIR }}}}/{out_loc}/{assay}.csv"
      smilesColumn: canonical_smiles
    driver: intake_rdkit.smiles.SmilesSource
    metadata:
'''
    for k,v in md['metadata'].items():
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
  split_column: split
  class_column: active
  activity_column: standard_value

sources:
'''
with open('chemblassay_SIMPD.yaml','w+') as outf:
    print(hdr+'\n'.join(assay_accum),file=outf)