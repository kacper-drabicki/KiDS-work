import pytest
import pandas as pd
from kids.catalog.training_catalog import clean_zwarn, rename_class, load_and_prepare_training_catalog, merge_training_catalogs

@pytest.mark.parametrize(
        'column, values, expected_rows',
        [
            ('ZWARN', [0, 1, 0, 2], 2),
            ('ZWARNING', [0, 1, 0, 2, 0], 3),
        ],
)
def test_clean_zwarn(column, values, expected_rows):
    df = pd.DataFrame({column: values})
    cleaned_df = clean_zwarn(df)
    
    assert len(cleaned_df) == expected_rows
    assert all(cleaned_df[column] == 0)

@pytest.mark.parametrize(
        'column',
        [
            ('CLASS'),
            ('SPECTYPE'),
        ],
)
def test_rename_class(column):
    df = pd.DataFrame({column: ['A', 'B', 'C']})
    renamed_df = rename_class(df)
    
    assert 'SPECTYPE' in renamed_df.columns
    assert 'CLASS' not in renamed_df.columns
    assert all(renamed_df['SPECTYPE'] == ['A', 'B', 'C'])

def test_merge_training_catalogs():
    df1 = pd.DataFrame({'ID': [1, 2, 3], 'Value': ['A', 'B', 'C']})
    df2 = pd.DataFrame({'ID': [3, 4, 5], 'Value': ['D', 'D', 'E']})
    
    merged_df = merge_training_catalogs(df1, df2)
    
    assert len(merged_df) == 5
    assert set(merged_df['ID']) == {1, 2, 3, 4, 5}
    assert merged_df.loc[merged_df['ID'] == 3, 'Value'].iloc[0] == 'C' # Keep first occurrence

def test_load_and_prepare_training_catalog():
    pass
