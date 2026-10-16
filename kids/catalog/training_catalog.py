import pandas as pd
from kids.catalog.catalog import load_catalog, clean_mask, replace_sentinel, add_colors, add_ratios

def clean_zwarn(df: pd.DataFrame) -> pd.DataFrame:
    zwarn_col = 'ZWARN' if 'ZWARN' in df else 'ZWARNING'

    return df.loc[df[zwarn_col] == 0]

def rename_class(df: pd.DataFrame) -> pd.DataFrame:
    ''' 
        SDSS and DESI catalogs have different names for the class column. 
        This function renames the column to a common name. 
    '''
    return df.rename(columns={'CLASS': 'SPECTYPE'}) # No error will be raised if the column does not exist, as we are using the rename method with a dictionary.

def load_and_prepare_training_catalog(file_path: str) -> pd.DataFrame:
    df = (
        load_catalog(file_path)
        .pipe(clean_mask)
        .pipe(clean_zwarn)
        .pipe(replace_sentinel)
        .pipe(add_colors)
        .pipe(add_ratios)
        .pipe(rename_class)
        .reset_index(drop=True, inplace=False) 
    )
    return df

def merge_training_catalogs(catalog_one: pd.DataFrame, catalog_two: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([catalog_one, catalog_two]).drop_duplicates(subset=['ID'], keep='first').reset_index(drop=True, inplace=False)

