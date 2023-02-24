import intake
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import numpy as np
import ga_lib_3 as ga_lib
import sys
import math

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score,cohen_kappa_score

def run_it_oob_optimization(oob_probs, labels_train, thresholds, ThOpt_metrics = 'Kappa'):
    """Optimize the decision threshold based on the prediction probabilities of the out-of-bag set of random forest.
    The threshold that maximizes the Cohen's kappa coefficient or a ROC-based criterion 
    on the out-of-bag set is chosen as optimal.
    
    Parameters
    ----------
    oob_probs : list of floats
        Positive prediction probabilities for the out-of-bag set of a trained random forest model
    labels_train: list of int
        True labels for the training set
    thresholds: list of floats
        List of decision thresholds to screen for classification
    ThOpt_metrics: str
        Optimization metric. Choose between "Kappa" and "ROC"
        
    Returns
    ----------
    thresh: float
        Optimal decision threshold for classification
    """
    # Optmize the decision threshold based on the Cohen's Kappa coefficient
    if ThOpt_metrics == 'Kappa':
        tscores = []
        # evaluate the score on the oob using different thresholds
        for thresh in thresholds:
            scores = [1 if x>=thresh else 0 for x in oob_probs]
            kappa = metrics.cohen_kappa_score(labels_train,scores)
            tscores.append((np.round(kappa,3),thresh))
        # select the threshold providing the highest kappa score as optimal
        tscores.sort(reverse=True)
        thresh = tscores[0][-1]
    # Optmize the decision threshold based on the ROC-curve
    elif ThOpt_metrics == 'ROC':
        # ROC optimization with thresholds determined by the roc_curve function of sklearn
        fpr, tpr, thresholds_roc = metrics.roc_curve(labels_train, oob_probs, pos_label=1)
        specificity = 1-fpr
        roc_dist_01corner = (2*tpr*specificity)/(tpr+specificity)
        thresh = thresholds_roc[np.argmax(roc_dist_01corner)]
    return thresh


def test_rf_model(X,y,nIters=10,numThreads=1):
    X = np.array(X)
    y = np.array(y)

    sss = StratifiedShuffleSplit(n_splits = nIters, test_size=0.2, random_state=0xf00d)
    scores = []
    kappas = []
    balanced = []
    rocs = []
    for train,test in sss.split(X,y):
        X_train,X_test = X[train],X[test]
        y_train,y_test = y[train],y[test]
        cls = RandomForestClassifier(n_estimators=500,max_depth=15,min_samples_leaf=2,min_samples_split=4,
                                     random_state=0xf00d,n_jobs=num_threads,oob_score=True)
        cls.fit(X_train,y_train)

        oobProbs = [x[1] for x in cls.oob_decision_function_]
        thresh = run_it_oob_optimization(oobProbs,y_train,np.arange(0.10,0.55,0.05))
        probs = cls.predict_proba(X_test)
        preds = [1 if x[1]>=thresh else 0 for x in probs]
        kappas.append(cohen_kappa_score(y_test,preds))
        balanced.append(balanced_accuracy_score(y_test,preds))
        scores.append(cls.score(X_test,y_test))

        auc = metrics.roc_auc_score(y_test, [x[1] for x in probs])
        rocs.append(auc)


    print(rocs)    
    return rocs,scores,kappas,balanced



import argparse

train_frac_active = -1


parser = argparse.ArgumentParser()
parser.add_argument("assay")
parser.add_argument("--numThreads", type=int, default=1)
parser.add_argument("--balanced","-b",default=False,action="store_true")
parser.add_argument("--baseline", default=False, action="store_true") # whether or not to use MFP0
args = parser.parse_args()

# Name of the dataset to use:
assay = args.assay
num_threads = args.numThreads
isBalanced = args.balanced
doBaseline = args.baseline

# -----------------------
# reading the molecules into a dataframe
# The rest of the setup code assumes that the data frame df has a column
# named 'mol' with rdkit molecules. If you don't have that, you could just use a list
# of molecules and adjust the code below to reflect that
catalog = intake.open_catalog('./datasets/public_data.yaml')
bioactivities = catalog.assays

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
if not isBalanced:
    upper, lower = ga_lib.get_imbalanced_bins(pActs,active_inactive_offset=0.0)
else:
    upper, lower = ga_lib.get_imbalanced_bins(pActs,tgt_frac=0.4,active_inactive_offset=0.0)

df = df[(pActs >= upper) | (pActs <= lower)]
binned = [1 if x >= upper else 0 for x in df.pActivity]
df['active'] = ['active' if x == 1 else 'inactive' for x in binned]

mols = [
    Chem.MolFromSmiles(tmp_smi)
    for tmp_smi in df['canonical_smiles'].to_numpy(dtype=str)
]

if not doBaseline:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    fps = [generator.GetFingerprintAsNumPy(x) for x in mols]
else:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=0)
    fps = [generator.GetCountFingerprintAsNumPy(x) for x in mols]


print('------------\nRunning RF experiments')
rocs,scores,kappas,balanced = test_rf_model(fps,binned,numThreads=num_threads)

import pickle
import gzip

outfn = './'
if doBaseline:
    outfn += 'mfp0_'
if not isBalanced:
    outfn += 'nogap_ml_results_imbalanced_assay'
else:
    outfn += 'nogap_ml_results_balanced_assay'

with gzip.open(
        f'{outfn}/{assay}_randomsplit.pkl.gz',
        'wb+') as outf:
    pickle.dump((rocs,scores,kappas,balanced), outf)
