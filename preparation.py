import pandas as pd
import numpy as np
import re
import unicodedata

def normalize_columns(columns):
    """
    Normalizes column names by converting to lowercase, 
    removing parentheses and accents, and shortening words.

    Parameters:
    - columns (iterable): An iterable of column names to be normalized.

    Returns:
    - list: A list of normalized column names.
    """
    new_cols = []

    for col in columns:
        col = col.lower()
        col = re.sub(r"\(.*?\)", "", col)
        col = unicodedata.normalize("NFKD", col)
        col = col.encode("ascii", "ignore").decode("utf-8")
        col = re.sub(r"[^a-z0-9]+", " ", col)
        words = col.split()
        short_words = [w[:3] for w in words]
        new_cols.append("_".join(short_words))
        
    return new_cols


def columns_selection(df):
    """
    Selects specific columns from the DataFrame based on their index positions.

    Parameters:
    - df (pd.DataFrame): The input DataFrame from which to select columns.

    Returns:
    - pd.DataFrame: A DataFrame containing only the selected columns.
    """
    cols_to_keep = [1, 2, 7, 8]

    return df.iloc[:, cols_to_keep]


def data_cleaning(df):
    """
    Cleans the input DataFrame by normalizing column names, selecting specific columns,
    and converting date and time columns to appropriate formats.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
    - pd.DataFrame: A cleaned DataFrame with normalized column names, selected columns,
      and properly formatted date and time columns.
    """
    df.columns = normalize_columns(df.columns)
    df = columns_selection(df).copy()
    
    df['timestamp'] = pd.to_datetime(df['dat'] + ' ' + df['heu'], format='%d/%m/%Y %H:%M')
    df['dat'] = pd.to_datetime(df['dat'], format='%d/%m/%Y').dt.date
    df['heu'] = pd.to_datetime(df['heu'], format='%H:%M').dt.time
    df = df.set_index('timestamp').sort_index()

    df = df.loc[df['con_bru_ele_rte'].notna()]

    return df

def create_dfs(df):
    df_gaz = df.drop(columns=['con_bru_ele_rte'])
    df_gaz = df_gaz.loc[df_gaz['con_bru_gaz_rte'].notna()]
    df_gaz = df.drop(columns=['con_bru_gaz_tot'])
    df_gaz = df_gaz.loc[df_gaz['con_bru_gaz_tot'].notna()]