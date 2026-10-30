import pandas as pd
from kids.catalog.loading import load_catalog
from kids.catalog.cleaning import clean_mask, clean_zwarn, replace_sentinel, rename_class
from kids.catalog.features import add_colors, add_ratios

def load_and_prepare_training_catalog(file_path: str) -> pd.DataFrame:
    df = (
        load_catalog(file_path, index_col=0)
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

def load_and_prepare_inference_catalog(file_path: str) -> pd.DataFrame:
    df = (
        load_catalog(file_path, index_col=None)
        .pipe(clean_mask)
        .pipe(replace_sentinel)
        .pipe(add_colors)
        .pipe(add_ratios)
        .reset_index(drop=True, inplace=False) 
    )
    return df

