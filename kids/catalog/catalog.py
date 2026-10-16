import pandas as pd
import numpy as np
from itertools import combinations
from kids.features import MAGNITUDES

def load_catalog(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, index_col=0)

def clean_mask(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df['MASK'] <= 1]

def replace_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([-99.0, 99.0], np.nan)

def add_colors(df: pd.DataFrame) -> pd.DataFrame:
    colors = [df[m1].sub(df[m2]).rename(f"{m1.split('_')[-1]}-{m2.split('_')[-1]}") 
              for m1, m2 in combinations(MAGNITUDES, 2)]
    return pd.concat([df] + colors, axis=1)

def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    ratios = [df[m1].div(df[m2]).rename(f"{m1.split('_')[-1]}/{m2.split('_')[-1]}") 
              for m1, m2 in combinations(MAGNITUDES, 2)]
    return pd.concat([df] + ratios, axis=1)
