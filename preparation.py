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


def add_holidays(df):
    """
    Add holidays from the french scholar calendar.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Return:
    - pd.DataFrame: A DataFrame containing holidays.
    """
    df_vacances = pd.read_csv(
        'DATA/fr-en-calendrier-scolaire.csv', 
        sep=';'
    )
    df_vacances.columns = normalize_columns(df_vacances.columns)

    df_vacances['dat_de_deb'] = pd.to_datetime(df_vacances['dat_de_deb']).dt.tz_localize(None).dt.normalize()
    df_vacances['dat_de_fin'] = pd.to_datetime(df_vacances['dat_de_fin']).dt.tz_localize(None).dt.normalize()
    mask = (
        (df_vacances['dat_de_fin'] >= df.index.min()) & 
        (df_vacances['pop'] != "Enseignants") &
        (df_vacances['zon'].isin(['Zone A', 'Zone B', 'Zone C', 'Corse']))
    )
    df_vacances = df_vacances.loc[mask]
    df_vacances = (
        df_vacances
        .drop(columns=['pop', 'aca'])
        .groupby(['des', 'ann_sco',  'dat_de_deb', 'dat_de_fin'], as_index=False)
        .agg({'dat_de_deb': 'min', 'dat_de_fin': 'max'})
        .sort_values('dat_de_deb')
    )

    noms_vac = [x for x in df_vacances['des'].unique() if str(x).startswith('Vacances')]
    for vac in noms_vac:
        df[vac] = 0
        vac_rows = df_vacances[df_vacances['des'] == vac]
        for _, row in vac_rows.iterrows():
            debut = row['dat_de_deb']  
            fin = row['dat_de_fin']
            mask = (df.index >= debut) & (df.index <= fin)
            df.loc[mask, vac] = 1

    new_cols = normalize_columns(df[noms_vac].columns)
    df.rename(columns=dict(zip(df[noms_vac].columns, new_cols)), inplace=True)

    return df


def add_bank_holidays(df):
    """
    Add french bank holidays.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Return:
    - pd.DataFrame: A DataFrame containing bank holidays.
    """
    df_ferie = pd.read_csv(
        'DATA/jours_feries_metropole.csv', 
        sep = ','
    )
    df_ferie['date'] = pd.to_datetime(df_ferie['date'])
    df['top_fer'] = df.index.normalize().isin(df_ferie['date']).astype(int)

    return df


def data_cleaning(df):
    """
    Cleans the input DataFrame by normalizing column names, 
    selecting specific columns,
    converting date and time columns to appropriate formats, 
    adds temporal features for time series analysis, 
    and adds scholar and bank holidays.

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

    df['day'] = df.index.day
    df['day_name'] = df.index.day_name()
    df['week'] = df.index.isocalendar().week
    max_week = 5
    df['week_of_month'] = ((df.index.day - 1) // 7) + 1
    df['week_of_month_sin'] = np.sin(2 * np.pi * df['week_of_month'] / max_week)
    df['week_of_month_cos'] = np.cos(2 * np.pi * df['week_of_month'] / max_week)
    df['month'] = df.index.month
    df['month_name'] = df.index.month_name()
    df['year'] = df.index.year

    df = add_holidays(df)
    df = add_bank_holidays(df)

    cols = [c for c in df.columns if c not in ['con_bru_gaz_tot', 'con_bru_ele_rte']]  # toutes sauf C et D
    new_order = cols + ['con_bru_gaz_tot', 'con_bru_ele_rte']  # mettre C et D à la fin
    df = df[new_order]  



    return df

def create_dfs(df):
    df_gaz = df.drop(columns=['con_bru_ele_rte'])
    df_gaz = df_gaz.loc[df_gaz['con_bru_gaz_tot'].notna()]
    
    df_ele = df.drop(columns=['con_bru_gaz_tot'])
    df_ele = df_ele.loc[df_ele['con_bru_ele_rte'].notna()]

    return df_gaz, df_ele