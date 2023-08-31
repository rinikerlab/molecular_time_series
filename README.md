# Citation

This repo contains the code and datasets for the publication

    G.A. Landrum, M. Beckers, J. Lanini, N. Schneider, N. Stiefl, S. Riniker 
    "SIMPD: an Algorithm for Generating Simulated Time Splits for Validating Machine Learning Approaches"
    https://chemrxiv.org/engage/chemrxiv/article-details/6406049e6642bf8c8f10e189


# Data and results

## Directories:
- datasets: contains the Ki datasets exported from ChEMBL32 as well as the results of running SIMPD on the assay datasets. There's also a data catalog description for the intake package in here

# Jupyter Notebooks
- 01_Get_ChEMBL32_Bioactivity_Data_assays.ipynb : export the raw ChEMBL Ki datasets from ChEMBL
- 02_ChEMBL_ML_experiments.ipynb : analysis of the results of the ML experiments on the ChEMBL Datasets

# Python files
- ga_lib_3.py : Python library with utility functionality for implementing SIMPD and doing the analysis. This is used in most of the scripts and juypter notebooks

# Other
- pymoo.yml : conda environment specification for the environment that was used during this project
- run_chemblassay_ml_nogap.py : script to run the ML experiments on the base assay data
- run_chemblassay_ga_nogap.py : script to run SIMPD on the base assay data
- run_chemblassay_ml_moo_nogap.py : script to run the ML experiments on the SIMPD results
- create_simpd_output_chemblKi.py : script to create CSV files with the final SIMPD datasets. This is what ends up in the datasets/chembl_SIMPD directory



