import pandas as pd
import numpy as np
from itertools import combinations
from kids.features import MAGNITUDES

def load_catalog(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, index_col=0)

def clean_mask(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df['MASK'] <= 1]

def clean_zwarn(df: pd.DataFrame) -> pd.DataFrame:
    zwarn_col = 'ZWARN' if 'ZWARN' in df else 'ZWARNING'

    return df.loc[df[zwarn_col] == 0]

def replace_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([-99.0, 99.0], np.nan)

def add_colors(df: pd.DataFrame) -> pd.DataFrame:
    colors = [df[m1].sub(df[m2]).rename(f"{m1.split('_')[-1]}-{m2.split('_')[-1]}") 
              for m1, m2 in combinations(MAGNITUDES, 2)]
    return pd.concat([df] + colors, axis=1)

def rename_class(df: pd.DataFrame) -> pd.DataFrame:
    ''' 
        SDSS and DESI catalogs have different names for the class column. 
        This function renames the column to a common name. 
    '''
    if 'CLASS' in df:
        df = df.rename(columns={'CLASS': 'SPECTYPE'})
    return df

def load_and_prepare_catalog(file_path: str) -> pd.DataFrame:
    df = (
        load_catalog(file_path)
        .pipe(clean_mask)
        .pipe(clean_zwarn)
        .pipe(replace_sentinel)
        .pipe(add_colors)
        .pipe(rename_class)
        .reset_index(drop=True) 
    )
    return df

def merge_catalogs(catalog_one: pd.DataFrame, catalog_two: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([catalog_one, catalog_two]).drop_duplicates(subset=['ID'], keep='first').reset_index(drop=True, inplace=False)

