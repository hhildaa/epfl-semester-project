# Semester project - Modeling migration intentions worldwide

## Abstract

The main goal is to cluster countries based on factors influencing migration aspiration and destination choice, to obtain more robust models for modeling the intentions and destination choices. For this aim, different methods are used, ranging from decision trees and regression with hierarchical methods and K-Means

## Requirements

There is a `requirements.txt` file in the repository, that lists the packages and their versions.

## Getting started

For running the data analysis files, one should have the Gallup World Poll dataset. As it is not a publicly available dataset, the dataset is not included in this repository. However, if one has the dataset in the folder structure indicated below, the result will be reproductible. As a consequence, none of the data is included in the `gwp_data` folder, but all can be made by having the zip file. The structure is included for transparency. Every other needed data is included in the repository. 

To attain the results, one should run the notebooks in the given orders. The resulting plots and lists will appear in these notebooks.

## Folder structure

    .
    ├── country_data
    │   └── country_attributes.xlsx                     # Country attributes dataset
    ├── gwp_data
    │   ├──Gallup_World_Poll_Wave_11_17_091622.zip      # The original dataset
    │   ├── clean_per_year
    │   │   └── ...                                     # Prepared dataset used in analysis, produced by the notebooks
    │   └── prepared_aspiration
    │   │   └── ...                                     # Prepared dataset used in analysis, produced by the notebooks
    │   └── prepared_destination
    │       └── ...                                     # Prepared dataset used in analysis, produced by the notebooks
    ├── meta
    │   ├── countrynum_to_name_dict                     # Pickle file, hand-made dictionary: maps country number to name
    │   ├── countrynum_to_ISO_dict                      # Pickle file, hand-made dictionary: maps country number to ISO code
    │   ├── columns.xlsx                                # Hand-made categorization of the columns of the GWP data
    │   └── columns                                     # Pickle file, made from the excel file
    │
    ├── 1_dataprep_1_columns.ipynb                      # Data preparation notebook: columns excel file
    ├── 1_dataprep_2_GWP.ipynb                          # Data preparation notebook: GWP data
    ├── 1_dataprep_3_country.ipynb                      # Data preparation notebook: country data
    ├── 2_EDA_country_data .ipynb                       # Exploratory data analysis on country data
    ├── 2_EDA_gwp_data.ipynb                            # Exploratory data analysis on GWP data
    ├── 3_clustering_origin.ipynb                       # Notebook for clustering migration origin
    ├── 4_decisiontree_aspiration.ipynb                 # Notebook for clustering with decision trees in the aspiration case
    ├── 4_decisiontree_USA.ipynb                        # Notebook for clustering with decision trees in the USA/not USA
    ├── 4_hierarchical_aspiration.ipynb                 # Notebook for clustering with hierarchical method in the aspiration case
    ├── 4_regression_destination.ipynb                  # Notebook for clustering destination choice
    │
    ├── cluster_methods.py                              # Help functions for clustering
    ├── cluster_vis.py                                  # Visualization function
    ├── dataprep.py                                     # Data preparation functions
    ├── decisiontree_help.py                            # Help functions for decision tree computations
    ├── hierarchical_help.py                            # Help functions for hierarchical clustering
    │
    ├── report.pdf                                      # Report
    ├── .gitignore
    └── README.md
