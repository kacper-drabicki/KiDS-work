import pandas as pd
import numpy as np

def clean_mask(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df['MASK'] <= 1]

def replace_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([-99.0, 99.0], np.nan)

def clean_zwarn(df: pd.DataFrame) -> pd.DataFrame:
    zwarn_col = 'ZWARN' if 'ZWARN' in df else 'ZWARNING'

    return df.loc[df[zwarn_col] == 0]

def rename_class(df: pd.DataFrame) -> pd.DataFrame:
    ''' 
        SDSS and DESI catalogs have different names for the class column. 
        This function renames the column to a common name. 
    '''
    # No error will be raised if the column does not exist, as we are using the rename method with a dictionary.
    return df.rename(columns={'CLASS': 'SPECTYPE'})   
