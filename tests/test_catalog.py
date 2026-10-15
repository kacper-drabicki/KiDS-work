import pandas as pd
import numpy as np
from kids.catalog.catalog import clean_mask, replace_sentinel, add_colors
from kids.features import MAGNITUDES, COLORS

def test_clean_mask():
   df = pd.DataFrame({'MASK': [0, 2, 1, 3]})
   cleaned_df = clean_mask(df)
   expected_df = pd.DataFrame({'MASK': [0, 1]}, index=[0, 2])

   assert cleaned_df.equals(expected_df)

def test_replace_sentinel():
   df = pd.DataFrame({'A': [1, -99.0, 3], 'B': [4, 5, 99.0]})
   replaced_df = replace_sentinel(df)
   expected_df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})

   assert replaced_df.equals(expected_df)

def test_add_colors():
    df = pd.DataFrame({magnitude: [20.0 - i] for i, magnitude in enumerate(MAGNITUDES)})
    df_with_colors = add_colors(df)

    assert df_with_colors.loc[0, 'u-g'] == 1.0
    assert all(color in df_with_colors.columns for color in COLORS)
    assert len(df_with_colors.columns) == len(MAGNITUDES) + len(COLORS)


