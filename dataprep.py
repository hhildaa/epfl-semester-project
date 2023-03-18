# imports
import os
import pandas as pd
import pyreadstat
import numpy as np
from zipfile import ZipFile
from typing import Tuple, Dict, Any

# def read_file(source_num) -> Tuple[pd.DataFrame, Dict[str, Any]]:
#     """
#     Reads the SAV file from the source zip archive, and returns it as a dataframe and a dictionary of metadata.
    
#     Returns:
#         A tuple containing the dataframe and the metadata dictionary.
#     """
#     if source_num == 1:
#         source = "gwp_data/Gallup_World_Poll_Wave_1_5_091622.zip"
#         num = '1_5_091622'
#     elif source_num == 2:
#         source = "gwp_data/Gallup_World_Poll_Wave_6_10_091622.zip"
#         num = '6_10_091622'
#     elif source_num == 3:
#         source = "gwp_data/Gallup_World_Poll_Wave_11_17_091622.zip"
#         num = '11_17_091622'

#     # save file to a temporary folder
#     with ZipFile(source, 'r') as zipObject:
#         listOfFileNames = zipObject.namelist()
#         for fileName in listOfFileNames:
#             if fileName.endswith('.sav'):
#                 # Extract a single file from zip
#                 if os.path.exists(f'temp_sav/{fileName}')==False:
#                     zipObject.extract(fileName, 'temp_sav')
#                     print('All the sav files are extracted')
#                     # open file
#                     df, meta = pyreadstat.read_sav(f'temp_sav/{fileName}') 
#                 else: 
#                     # open file
#                     df, meta = pyreadstat.read_sav(f'temp_sav/{fileName}') 

#     # rename the columns to more readable names
#     df.columns=meta.column_labels

#     print('Data is saved in df and meta variables with readable column names')
#     return df, meta

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

def remove_aspiration_DK(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data preparation: MIGRATION ASPIRATION
    Removes rows where the 'WP1325: Move Permanently to Another Country' column is missing,
     or has a value other than 1 or 2 (Yes/No). Also removes the 'WP3120: Country Would Move To' column.
    
    Args:
        df: The dataframe to modify.
    
    Returns:
        The modified dataframe with the unwanted rows and columns removed.
    """
    df.convert_dtypes()
    df = df[df['WP1325: Move Permanently to Another Country'].notnull()]
    df.drop(['WP3120: Country Would Move To'], axis=1, inplace=True)
    df = df[(df['WP1325: Move Permanently to Another Country']==1) | (df['WP1325: Move Permanently to Another Country']==2)]
    
    print("Only kept yes or no answers in aspiration.")
    return df   

def sampling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Samples observations from the dataframe so that each country has the same number 
    of observations. The number of observations per country is equal to the mean of 
    the value counts for the 'WP5: Country' column.
    
    Args:
        df: The dataframe to sample from.
    
    Returns:
        The modified dataframe with the sampled observations.
    """
    n = df["WP5: Country"].value_counts().mean()
    df = df.groupby("WP5: Country").sample(int(n), replace = True, random_state = 2022)
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
        if col != "WP5: Country" :
            if col!='WP1325: Move Permanently to Another Country':
                if col!='WP3120: Country Would Move To':
                    quest = 0 in df[["WP5: Country", col]].groupby("WP5: Country").count().values.flatten() 
                    if quest == True: 
                        columns_with_missing.append(col)

    df.drop(columns_with_missing, axis=1, inplace=True)
    print('Questions only asked in specific countries are removed.')
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

    # impute missing values, 
    # df["Region"] = df[list(set(regions).intersection(df.columns))].sum(axis=1, min_count=1)

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
    # meta.column_names_to_labels["Region"] = "Region"

    # rename the columns to more readable names
    df.columns=(val+ ": " + meta.column_names_to_labels[val] for val in df.columns)
 
    print("Codes added to the feature names.")
    return df

def impute_missing_by_type(df: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in the input dataframe based on the column types.
     The column types are inferred from the input metadata dataframe. 
     For Yes/No questions, missing values are replaced with 3 (Don't know/Refused).
     For Ordinal and Yes questions, missing values are replaced with the mode or mean 
     of the column, respectively. For No questions, a new binary column is created indicating
     the presence of missing values.
    
    Args:
        df: The dataframe whose missing values are to be imputed.
        df_meta: The metadata dataframe that contains the information about the column types.
    
    Returns:
        The modified dataframe with imputed missing values.
    """
    df = df.convert_dtypes()

    # sorting by column type
    yes_columns = []
    yn_columns = []
    ordinal_columns = []
    no_columns = []
    other = []

    for col in df.columns:
        l = list(df_meta[df_meta['column'].str.contains(col)]["categorical?"])
        if len(l) !=0:
            if "yes" in l[0]:
                yes_columns.append(col)
            elif "yn" in l[0] :
                yn_columns.append(col)
            elif "ordinal" in l[0] :
                ordinal_columns.append(col)
            elif "no" in l[0] :
                no_columns.append(col)
            else:
                other.append(col)
        else:
            other.append(col)

    for col in other:
        length = df[col].nunique()
        if length <= 3:
            yn_columns.append(col)
        elif length <= 10:
            ordinal_columns.append(col)
        else:
            no_columns.append(col)
    
    # imputing by column type
    for col in yn_columns: 
        df[[col]].replace({4: 3, pd.NA: 4}, inplace=True)

    for col in yes_columns: 
        df[[col]].replace({pd.NA: 3}, inplace=True)

    for col in no_columns:
        df[str(col)+"_missing"] = np.where(df[col].isnull(), 1, 0)
        if col == ['WP1220: Age'] or col == ['Age']:
            df[[col]].replace({101: pd.NA}, inplace=True)
        df[col].fillna(int(df[col].mean()), inplace=True)

    for col in ordinal_columns: 
        df[str(col)+"_missing"] = np.where(df[col].isnull(), 1, 0)
        df[col].fillna(int(df[col].mode()), inplace=True)

    print("Missing values are imputed.")
    return df

def preprocess_gwp_asp(df, meta, df_meta):

    # Regions
    df = new_region_column(df)

    # Renaming the features
    df = rename_with_codes(df, meta)

    # Removing unwanted columns
    df = remove_unwanted(df, df_meta)

    # Remove questions which are not asked in all countries. 
    df = remove_notallcountry(df)
    try:
        df.drop('INDEX_CA: Community Attachment Index', axis=1, inplace = True)
    except:
        pass
    
    return df