# imports
import os
from typing import Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd


def preprocess_gwp_asp_wrangling(df: pd.DataFrame, meta, df_meta: pd.DataFrame):
    # Regions
    df = new_region_column(df)

    # Renaming the features
    df = rename_with_codes(df, meta)

    # Removing unwanted columns
    df = remove_unwanted(df, df_meta)

    # Remove questions which are not asked in all countries.
    df = remove_notallcountry(df)
    try:
        df.drop("INDEX_CA: Community Attachment Index", axis=1, inplace=True)
    except:
        pass

    return df


def new_region_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new 'Region' column in the dataframe by summing the values of all columns in the input dataframe that are listed in the 'meta/regions.txt' file. Missing values are imputed.

    Args:
        df: The dataframe to modify.

    Returns:
        The modified dataframe with the new 'Region' column.
    """
    regions = open("meta/regions.txt").readlines()
    regions = list([text[1:-3] for text in regions])

    # drop the unwanted hand-picked columns
    drop_cols = list(set(df.columns).intersection(set(regions)))
    df.drop(drop_cols, axis=1, inplace=True)

    print("Regions are imputed in one column.")
    return df


def rename_with_codes(df: pd.DataFrame, meta) -> pd.DataFrame:
    """
    Renames the columns of the input dataframe using the corresponding labels from
     the input metadata object. The codes for the columns are added to the labels,
     separated by a colon.

    Args:
        df: The dataframe whose columns are to be renamed.
        meta: The metadata object that contains the column labels.

    Returns:
        The dataframe with the renamed columns.
    """

    # rename the columns to more readable names
    df.columns = (val + ": " + meta.column_names_to_labels[val] for val in df.columns)

    print("Codes added to the feature names.")
    return df


def remove_unwanted(df: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Removes technical columns and questions related to few countries from the dataframe,
     using the metadata dataframe to identify the columns to remove.

    Args:
        df: The dataframe to remove columns from.
        df_meta: The metadata dataframe, containing the "column" and "remove" columns. (Prepared by hand.)

    Returns:
        The modified dataframe with the unwanted columns removed.
    """
    remove = set(df_meta[df_meta["remove"] == "yes"]["column"].values)
    remove_full = list(remove.intersection(set(df.columns)))
    df.drop(remove_full, axis=1, inplace=True)

    print("Unwanted columns are removed.")
    return df


def remove_notallcountry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes columns from the dataframe that are not asked in all countries.
     The 'WP5: Country' column, the 'WP1325: Move Permanently to Another Country' column,
     and the 'WP3120: Country Would Move To' column are not removed.

    Args:
        df: The dataframe to modify.

    Returns:
        The modified dataframe with the unwanted columns removed.
    """
    columns_with_missing = []
    for col in df.columns:
        if col != "WP5: Country":
            if col != "WP1325: Move Permanently to Another Country":
                if col != "WP3120: Country Would Move To":
                    quest = (
                        0
                        in df[["WP5: Country", col]]
                        .groupby("WP5: Country")
                        .count()
                        .values.flatten()
                    )
                    if quest == True:
                        columns_with_missing.append(col)

    df.drop(columns_with_missing, axis=1, inplace=True)
    print("Questions only asked in specific countries are removed.")
    return df
