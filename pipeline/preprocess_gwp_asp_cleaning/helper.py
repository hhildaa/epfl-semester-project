import numpy as np
import pandas as pd


def preprocess_gwp_asp_cleaning(
    df: pd.DataFrame, df_meta: pd.DataFrame
) -> pd.DataFrame:
    if df["WP1325: Move Permanently to Another Country"].isnull().all():
        print("No migration aspiration data in this year.")
        return pd.DataFrame()

    # Check for duplicate columns and keep only the columns that are not duplicates
    duplicated_columns = df.columns.duplicated()
    df = df.loc[:, ~duplicated_columns]

    # remove lines where the answer is not yes or no
    df = remove_aspiration_DK(df)

    # impute values by type
    df = impute_missing_by_type(df, df_meta)

    # drop rows with missing values
    df.dropna(inplace=True)

    # sample the data so that each country has the same number of observations
    df = sampling(df)

    # reindex the dataframe
    df["index"] = range(len(df["WP1220: Age"]))
    df.set_index("index", inplace=True)

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
    df = df[df["WP1325: Move Permanently to Another Country"].notnull()]
    df.drop(["WP3120: Country Would Move To"], axis=1, inplace=True)
    df = df[
        (df["WP1325: Move Permanently to Another Country"] == 1)
        | (df["WP1325: Move Permanently to Another Country"] == 2)
    ]

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
    df = df.groupby("WP5: Country").sample(int(n), replace=True, random_state=2022)
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
        l = list(df_meta[df_meta["column"].str.contains(col)]["categorical?"])
        if len(l) != 0:
            if "yes" in l[0]:
                yes_columns.append(col)
            elif "yn" in l[0]:
                yn_columns.append(col)
            elif "ordinal" in l[0]:
                ordinal_columns.append(col)
            elif "no" in l[0]:
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
        if df[col].isnull().sum() == len(df[col]):
            continue
        df[str(col) + "_missing"] = np.where(df[col].isnull(), 1, 0)
        if col == ["WP1220: Age"] or col == ["Age"]:
            df[[col]].replace({101: pd.NA}, inplace=True)
        # if column is not empty
        df[col].fillna(int(df[col].mean()), inplace=True)

    for col in ordinal_columns:
        if df[col].isnull().sum() == len(df[col]):
            continue
        df[str(col) + "_missing"] = np.where(df[col].isnull(), 1, 0)
        df[col].fillna(int(df[col].mode()), inplace=True)

    print("Missing values are imputed.")
    return df
