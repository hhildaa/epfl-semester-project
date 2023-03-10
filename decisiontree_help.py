# This file includes help functions for the aspiration clustering.
# source for the tree_to_code function: https://mljar.com/blog/extract-rules-decision-tree/
# the other functions may also rely on this implementation

from sklearn.tree import _tree
from sklearn import tree
import numpy as np
import pandas as pd
import pickle

from typing import Union, List, Tuple, Dict
from sklearn.tree import DecisionTreeClassifier

import cluster_methods
import decisiontree_help
import cluster_vis

with open('meta/countrynum_to_ISO_dict.pickle', 'rb') as fp:
    countrynum_to_ISO_dict = pickle.load(fp)

with open('meta/countrynum_to_name_dict.pickle', 'rb') as fp:
    countrynum_to_name_dict = pickle.load(fp)

def tree_to_code(tree: _tree.Tree, feature_names: List[str]) -> None:
    """
    Converts a decision tree object into an equivalent Python function.
     The resulting function can be used to make predictions based on the trained model.
     The function is printed to the standard output.
    
    Args:
        tree: The decision tree object to convert.
        feature_names: The names of the features used by the decision tree.
    
    Returns:
        None. The function is printed to the standard output.
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    feature_names = [f.replace(" ", "_")[:-5] for f in feature_names]
    print("def predict():")

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if '{}' <= '{}':".format(indent, name, np.round(threshold,2)))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if '{}' > '{}'".format(indent, name, np.round(threshold,2)))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return '{}'".format(indent, tree_.value[node]))

    recurse(0, 1)

def help_clustering(tree: DecisionTreeClassifier, X: pd.DataFrame, df_original: pd.DataFrame, depth: int) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    '''
    This function helps to visualize the tree and its clusters.
    The data is divided into two or three clusters based on the decision tree.

    Args:
        tree: DecisionTreeClassifier object
        X: DataFrame of features
        df_original: original DataFrame
        depth: integer between 2 and 3

    Returns:
        If depth is 2, returns Tuple of 4 DataFrames,
        If depth is 3, returns Tuple of 8 DataFrames
'''
    tree_ = tree.tree_

    # first decision
    left = df_original[df_original[X.columns[tree_.feature[0]]] <= tree_.threshold[0]]
    right = df_original[df_original[X.columns[tree_.feature[0]]] > tree_.threshold[0]]

    # second decision
    ll = left[left[X.columns[tree_.feature[1]]] <= tree_.threshold[1]]
    lr = left[left[X.columns[tree_.feature[1]]] > tree_.threshold[1]]
    rl = right[right[X.columns[tree_.feature[4]]] <= tree_.threshold[4]]
    rr = right[right[X.columns[tree_.feature[4]]] > tree_.threshold[4]]

    # third decision
    lll = ll[ll[X.columns[tree_.feature[2]]] <= tree_.threshold[2]]
    llr = ll[ll[X.columns[tree_.feature[2]]] > tree_.threshold[2]]
    lrl = lr[lr[X.columns[tree_.feature[3]]] <= tree_.threshold[3]]
    lrr = lr[lr[X.columns[tree_.feature[3]]] > tree_.threshold[3]]
    rll = rl[rl[X.columns[tree_.feature[5]]] <= tree_.threshold[5]]
    rlr = rl[rl[X.columns[tree_.feature[5]]] > tree_.threshold[5]]
    rrl = rr[rr[X.columns[tree_.feature[6]]] <= tree_.threshold[6]]
    rrr = rr[rr[X.columns[tree_.feature[6]]] > tree_.threshold[6]]

    if depth == 2:
        return ll, lr, rl, rr
    if depth == 3:
        return lll, llr, lrl, lrr, rll, rlr, rrl, rrr

def clustering1_depth2(tree: DecisionTreeClassifier, 
                       X: pd.DataFrame, 
                       df_original: pd.DataFrame) -> pd.DataFrame:
    '''
    This function clusters countries in 4 groups based on their feature importance in the decision tree.
    The first 2 levels of the decision tree are used to create the 4 groups.
    The function returns a DataFrame containing the number of countries in each group and
    the reasons for the classification.

    Args:
        tree: decision tree classifier object 
        X: dataframe containing the feature values for each country
        df_original: dataframe containing the country names

    Returns:
        df_sol: dataframe containing the clusters for each country.
    '''
    leftleft, leftright, rightleft, rightright = help_clustering(tree, X, df_original, 2)
    LL = leftleft.groupby("WP5: Country").size()
    LR = leftright.groupby("WP5: Country").size()
    RL = rightleft.groupby("WP5: Country").size()
    RR = rightright.groupby("WP5: Country").size()

    df_sol = pd.DataFrame()
    df_sol["LL"] = LL
    df_sol["LR"] = LR
    df_sol["RL"] = RL
    df_sol["RR"] = RR

    df_sol["cluster_1"] = df_sol[["LL", "LR", "RL", "RR"]].idxmax(axis=1)
    df_sol["reason_1"] = df_sol[["LL", "LR", "RL", "RR"]].max(axis=1)/df_sol[["LL", "LR", "RL", "RR"]].sum(axis=1)

    # clusters by name
    c_names = ["LL", "LR", "RR", "RL"]
    with open('meta/countrynum_to_name_dict', 'rb') as fp:
        countrynum_to_name_dict = pickle.load(fp)

    countryname_cluster_dict = {countrynum_to_name_dict[elem]:cluster for (elem, cluster) in zip(list(df_sol.index), list(df_sol.cluster_1))}

    for c in c_names:
        print(f"======= cluster: {c} =======")
        for country, cluster in countryname_cluster_dict.items():
            if cluster == c:
                print(f"{country}")

    return df_sol

def clustering1_depth3(tree: DecisionTreeClassifier, 
                       X: pd.DataFrame, 
                       df_original: pd.DataFrame, 
                       countrynum_to_name_dict: Dict[int, str]) -> pd.DataFrame:
    '''
    This function clusters countries in 4 groups based on their feature importance in the decision tree.
    The first 2 levels of the decision tree are used to create the 4 groups.
    The function returns a DataFrame containing the number of countries in each group and
    the reasons for the classification.

    Args:
        tree: decision tree classifier object 
        X: dataframe containing the feature values for each country
        df_original: dataframe containing the country names
        countrynum_to_name_dict: dictionary mapping country numbers to country names

    Returns:
        df_sol: dataframe containing the clusters for each country.
    '''

    lll, llr, lrl, lrr, rll, rlr, rrl, rrr = help_clustering(tree, X, df_original, 3)
    df_sol = pd.DataFrame()
    df_sol["LLL"] = lll.groupby("WP5: Country").size()
    df_sol["LLR"] = llr.groupby("WP5: Country").size()
    df_sol["LRL"] = lrl.groupby("WP5: Country").size()
    df_sol["LRR"] = lrr.groupby("WP5: Country").size()
    df_sol["RLL"] = rll.groupby("WP5: Country").size()
    df_sol["RLR"] = rlr.groupby("WP5: Country").size()
    df_sol["RRL"] = rrl.groupby("WP5: Country").size()
    df_sol["RRR"] = rrr.groupby("WP5: Country").size()

    c_names = ["LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR"]

    df_sol["cluster_1"] = df_sol[c_names].idxmax(axis=1)
    df_sol["reason_1"] = df_sol[c_names].max(axis=1)/df_sol[c_names].sum(axis=1)

    # clusters by name
    print_clusters(countrynum_to_name_dict, c_names, list(df_sol["cluster_1"]), df_sol.index)

    return df_sol

def clustering2(df_sol: pd.DataFrame, depth: int, countrynum_to_name_dict: Dict[int, str]) -> pd.DataFrame: 
    """
    Performs clustering on the dataframe using the specified depth.

    Args:
        df_sol: The dataframe containing the data to cluster.
        depth: The depth of the clustering algorithm to use.
        countrynum_to_name_dict: A dictionary mapping country numbers to names.

    Returns:
        pandas.DataFrame: The updated dataframe with the added cluster information.
    """
    if depth == 2: 
        c_names = ["LL", "LR", "RL", "RR"]
    if depth == 3:
        c_names = ["LLL", "LLR", "LRL", "LRR", "RLL", "RLR", "RRL", "RRR"]
    df_sol["cluster_2"] = df_sol[c_names].apply(lambda s: str(s.abs().nlargest(2).index.tolist()[0] + s.abs().nlargest(2).index.tolist()[1]), axis=1)
    df_sol["reason_2"] = df_sol[c_names].apply(lambda s: s.abs().nlargest(2).tolist()[1], axis=1)

    clusters = df_sol["cluster_2"].value_counts()
    cluster_names = clusters.index.tolist()
    country = list(df_sol.index)
    print_clusters(countrynum_to_name_dict, cluster_names, list(df_sol.cluster_2), country)
    
    return df_sol

def print_clusters(countrynum_to_name_dict: Dict[int, str],
                   cluster_names: List[str], 
                   sol: List[str], 
                   country: List[int]) -> None:
    """
    Prints the list of countries belonging to each cluster in cluster_names.

    Args:
    - countrynum_to_name_dict: A dictionary that maps country numbers to country names.
    - cluster_names: A list of cluster names.
    - sol: A list of cluster names, where each element represents the cluster
           a country belongs to.
    - country: A list of country numbers, where each element represents a country.

    Returns:
    - None
    """
    try:
        countryname_cluster_dict = {countrynum_to_name_dict[elem]:cluster for (elem, cluster) in zip(country, sol)}
    except:
        countryname_cluster_dict = {elem:cluster for (elem, cluster) in zip(country, sol)}

    for c in cluster_names:
        print(f"======= cluster: {c} =======")
        for country, cluster in countryname_cluster_dict.items():
            if cluster == c:
                print(country)

def run_clusters_distribution(depth: Union[int, str], 
                              X: np.ndarray, 
                              Y: np.ndarray, 
                              df_original: pd.DataFrame, 
                              algo_type: str, 
                              param: float) -> None:
    """
    Trains a decision tree classifier, clusters the leaves of the tree using a specified algorithm,
     and visualizes the results.

    Args:
    - depth: The maximum depth of the decision tree. If this is an integer, the decision tree 
             will be trained with that maximum depth.
             If this is a string, the decision tree will be trained without a specified maximum depth.
    - X: The feature values for the training data, as a NumPy array.
    - Y: The target values for the training data, as a NumPy array.
    - df_original: The original dataframe containing the training data, 
             with columns "WP5: Country" and "COUNTRY_ISO3: Country ISO alpha-3 code".
    - algo_type: The type of clustering algorithm to use.
    - param: The parameter for the clustering algorithm (e.g. the number of clusters for k-means clustering).

    Returns:
    - None
    """

    df_help = create_df(depth, X, Y, df_original)

    cluster = cluster_methods.simple_cluster(df_help, param, algo_type)
    c_names = list(set(cluster))
    decisiontree_help.print_clusters(countrynum_to_name_dict, c_names, cluster, df_help.index)

    df_vis = pd.DataFrame()
    df_vis["WP5: Country"] = [countrynum_to_name_dict[c_code] for c_code in df_help.index]
    df_vis["COUNTRY_ISO3: Country ISO alpha-3 code"] = [countrynum_to_ISO_dict[c_code] for c_code in df_help.index]
    df_vis["cluster"] = cluster

    cluster_vis.cluster_visualization(df_vis, cluster, f"D{algo_type}_param{param}")

def create_df(depth: Union[int, str], X: np.ndarray, Y: np.ndarray,df_original: pd.DataFrame)  -> pd.DataFrame:
    '''
    Creates a dataframe to perform clustering on the distributions of countires in the decision tree leaves.
    Args:
    - depth: The maximum depth of the decision tree. If this is an integer, the decision tree 
             will be trained with that maximum depth.
             If this is a string, the decision tree will be trained without a specified maximum depth.
    - X: The feature values for the training data, as a NumPy array.
    - Y: The target values for the training data, as a NumPy array.
    - df_original: The original dataframe containing the training data, 
             with columns "WP5: Country" and "COUNTRY_ISO3: Country ISO alpha-3 code".
    Return:
    - df_help: prepared dataframe  for clustering on the leaf distributions.
    '''
    if type(depth) is int:
        clf = tree.DecisionTreeClassifier(max_depth=depth)
    else: 
        clf = tree.DecisionTreeClassifier()

    clf = clf.fit(X, Y)
    leaf = clf.apply(X)

    print("The prediction accuracy is:")
    print(sum(clf.predict(X) == Y) / len(Y))

    df_help = pd.DataFrame()
    df_help["leaf"] = leaf
    df_help["country"] = df_original["WP5: Country"]
    df_help.dropna(inplace=True)
    df_help = df_help.groupby(df_help.columns.tolist(),as_index=False).size()
    df_help = df_help.pivot(index="country", columns="leaf", values="leaf")
    df_help = df_help.fillna(0)
    df_help = df_help.div(df_help.sum(axis=1), axis=0)

    return df_help
    