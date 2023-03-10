# This file includes the help functions to run different clustering methods.

import pandas as pd
import numpy as np
from typing import Union

from sklearn.preprocessing import MinMaxScaler
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import KElbowVisualizer

SEED = 2022

def DBSCAN_algo(data: pd.DataFrame, param: float) -> np.ndarray:
    """
    Clusters the given data using the DBSCAN algorithm with the specified parameter.
    
    Args:
        data: The data to cluster, represented as a pandas DataFrame.
        param: The DBSCAN eps parameter, which determines the maximum distance between
         two samples for them to be considered part of the same cluster.
    
    Returns:
        An array of cluster labels for each sample in the data.
    """
    my_dbscan = DBSCAN(eps=param, min_samples=2, metric = 'l2').fit(data)
    return my_dbscan.labels_

def agglo_algo(data: pd.DataFrame, param: int) -> np.ndarray:
    """
    Clusters the given data using the Agglomerative Clustering algorithm with the specified parameter.
    
    Args:
        data: The data to cluster, represented as a pandas DataFrame.
        param: The number of clusters to create.
    
    Returns:
        An array of cluster labels for each sample in the data.
    """
    my_agglo = AgglomerativeClustering(n_clusters = param, linkage='ward').fit(data)
    return my_agglo.labels_


def KMeans_algo(data: pd.DataFrame, param: int) -> np.ndarray:
    """
    Clusters the given data using the K-Means algorithm with the specified parameter.
    
    Args:
        data: The data to cluster, represented as a pandas DataFrame.
        param: The number of clusters to create.
    
    Returns:
        An array of cluster labels for each sample in the data.
    """
    my_kmeans = KMeans(n_clusters=param, random_state=SEED)
    my_kmeans.fit(data)
    clusters = my_kmeans.predict(data)
    return clusters

def KModes_algo(data: pd.DataFrame, param: int) -> np.ndarray:
    """
    Clusters the given data using the K-Modes algorithm with the specified parameter.
    
    Args:
        data: The data to cluster, represented as a pandas DataFrame.
        param: The number of clusters to create.
    
    Returns:
        An array of cluster labels for each sample in the data.
    """
    kmodes = KModes(n_clusters = param, random_state=SEED)
    clusters = kmodes.fit_predict(data)
    return clusters

def silhouette(df: pd.DataFrame, mini: int, maxi: int, method: str = "kmeans") -> None:
    """
    Plots the silhouette scores for the given dataframe using the specified clustering algorithm and range of parameters.
    
    Args:
        df: The dataframe to cluster and plot the silhouette scores for.
        mini: The minimum parameter value to use for the clustering algorithm.
        maxi: The maximum parameter value to use for the clustering algorithm.
        method: The clustering algorithm to use. Can be "kmeans", "kmodes", "agglo", or "dbscan".
    
    Returns:
        None
    """
    for param in range(mini,maxi+1):
        if method == "kmeans":
            model = KMeans(param, random_state=SEED)
        if method == "kmodes":
            model = KMeans(param, random_state=SEED)
        if method == "agglo": 
            model = AgglomerativeClustering(param, linkage='ward', random_state=SEED)
        if method == "dbscan":
            model = DBSCAN(param, min_samples=2, metric = 'l2', random_state=SEED)
        
        visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
        visualizer.fit(df)       
        visualizer.show()   

def elbow_method(df: pd.DataFrame, mini: int, maxi: int, method: str = "kmeans") -> None:
    """
    Plots the elbow curve for the given dataframe using the specified clustering algorithm and range of parameters.
    
    Args:
        df: The dataframe to cluster and plot the elbow curve for.
        mini: The minimum parameter value to use for the clustering algorithm.
        maxi: The maximum parameter value to use for the clustering algorithm.
        method: The clustering algorithm to use. Can be "kmeans", "kmodes", or "agglo".
    
    Returns:
        None
    """
    if method == "kmeans":
        model = KMeans(random_state=SEED)
    if method == "kmodes":
        model = KMeans(random_state=SEED)
    if method == "agglo": 
        model = AgglomerativeClustering(linkage='ward')
    visualizer = KElbowVisualizer(model, k=(mini,maxi))
    visualizer.fit(df)
    visualizer.show()     

def simple_cluster(df: pd.DataFrame, param: Union[int, float], method: str = "kmeans") -> np.ndarray:
    """
    Clusters the given dataframe using the specified clustering algorithm and parameter.
    
    Args:
        df: The dataframe to cluster.
        param: The parameter to use for the clustering algorithm. The type of this parameter depends on the algorithm. For example, it can be an integer for K-Means or Agglomerative Clustering, or a float for DBSCAN.
        method: The clustering algorithm to use. Can be "kmeans", "dbscan", or "agglo".
    
    Returns:
        An array of cluster labels for each sample in the data.
    """
    if method == "kmeans":
        clusters = KMeans_algo(df, param)
    if method == "dbscan": 
        clusters = DBSCAN_algo(df, param)  
    if method == "agglo": 
        clusters = agglo_algo(df, param)
    return clusters

def dummy_cluster(df: pd.DataFrame, aggreg: str = "mode", K: int =4) -> np.ndarray:
    """
    Clusters the given dataframe using the specified aggregation method and the K-Means or K-Modes algorithms.
    
    Args:
        df: The dataframe to cluster.
        aggreg: The aggregation method to use. Can be "mean" or "mode".
        K: number of clusters
    
    Returns:
        An array of cluster labels for each sample in the data.
    """

    if aggreg=="mean":
        clusters = KMeans_algo(mean_aggregation(df), K)

    if aggreg == "mode": 

        clusters = KModes_algo(mode_aggregation(df), K)

    return clusters

def mean_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the mean value of each numeric column in the given DataFrame, grouped by 'WP5: Country'.
    The resulting DataFrame is then scaled using a MinMaxScaler and returned as a NumPy array.

    Args:
    df: The input DataFrame. 

    Returns:
        The scaled mean values of each numeric column in the input DataFrame, grouped by 'WP5: Country'.
    """
    df_mean = df.groupby("WP5: Country").mean()
    df_mean = df_mean.dropna(axis=1) 
    # scaling
    scaler = MinMaxScaler()
    df_norm = scaler.fit_transform(df_mean)
    try:
        df_norm.drop("WP5: Country", axis =1, inplace=True)
    except:
        pass
    return df_norm

def mode_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the mode value of each column in the given DataFrame, grouped by 'WP5: Country'.
    The resulting DataFrame is then returned.

    Args:
        df: The input DataFrame. 

    Returns:
        The mode values of each column in the input DataFrame, grouped by 'WP5: Country'.
    """
    df_mode = df.groupby(by="WP5: Country", as_index="WP5: Country").apply(lambda x: x.mode().iloc[0])
    df_mode = df_mode.dropna(axis=1)
    try:
        df_mode.drop("WP5: Country", axis =1, inplace=True)
    except:
        pass
    return df_mode