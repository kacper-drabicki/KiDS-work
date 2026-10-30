import pandas as pd
from itertools import combinations

MAGNITUDES = ['MAG_GAAP_u','MAG_GAAP_g', 'MAG_GAAP_r','MAG_GAAP_i1', 'MAG_GAAP_i2',
              'MAG_GAAP_Z', 'MAG_GAAP_Y','MAG_GAAP_J', 'MAG_GAAP_H','MAG_GAAP_Ks']

COLORS = ['u-g', 'u-r', 'u-i1', 'u-i2', 'u-Z', 'u-Y','u-J', 'u-H', 'u-Ks', 'g-r', 'g-i1', 'g-i2',
          'g-Z', 'g-Y', 'g-J', 'g-H','g-Ks', 'r-i1', 'r-i2', 'r-Z', 'r-Y', 'r-J', 'r-H', 'r-Ks', 
          'i1-i2','i1-Z', 'i1-Y', 'i1-J', 'i1-H', 'i1-Ks', 'i2-Z', 'i2-Y', 'i2-J', 'i2-H','i2-Ks', 
          'Z-Y', 'Z-J', 'Z-H', 'Z-Ks', 'Y-J', 'Y-H', 'Y-Ks', 'J-H','J-Ks', 'H-Ks']

RATIOS = ['u/g', 'u/r', 'u/i1', 'u/i2', 'u/Z', 'u/Y', 'u/J','u/H', 'u/Ks', 'g/r', 'g/i1', 'g/i2',
          'g/Z', 'g/Y', 'g/J', 'g/H', 'g/Ks', 'r/i1', 'r/i2', 'r/Z', 'r/Y', 'r/J', 'r/H', 'r/Ks',
          'i1/i2', 'i1/Z', 'i1/Y', 'i1/J', 'i1/H', 'i1/Ks', 'i2/Z', 'i2/Y', 'i2/J', 'i2/H', 'i2/Ks',
          'Z/Y', 'Z/J', 'Z/H', 'Z/Ks', 'Y/J', 'Y/H', 'Y/Ks', 'J/H', 'J/Ks', 'H/Ks']

CLF_FEATURES = COLORS + RATIOS + ['CLASS_STAR']
PHOTOZ_FEATURES = MAGNITUDES + COLORS

def add_colors(df: pd.DataFrame) -> pd.DataFrame:
    colors = [df[m1].sub(df[m2]).rename(f"{m1.split('_')[-1]}-{m2.split('_')[-1]}") 
              for m1, m2 in combinations(MAGNITUDES, 2)]
    return pd.concat([df] + colors, axis=1)

def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    ratios = [df[m1].div(df[m2]).rename(f"{m1.split('_')[-1]}/{m2.split('_')[-1]}") 
              for m1, m2 in combinations(MAGNITUDES, 2)]
    return pd.concat([df] + ratios, axis=1)
