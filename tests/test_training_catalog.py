import pytest
import pandas as pd
from kids.catalog.cleaning import clean_zwarn, rename_class
from kids.catalog.preparation import load_and_prepare_training_catalog, merge_training_catalogs
from kids.catalog.features import MAGNITUDES

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
    
    expected_df = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'Value': ['A', 'B', 'C', 'D', 'E'],
    })

    pd.testing.assert_frame_equal(merged_df, expected_df)
    assert merged_df.index.equals(pd.RangeIndex(5))


def test_load_and_prepare_training_catalog(tmp_path):
    catalog = pd.DataFrame({
        'ID': ['kept', 'bad-mask', 'bad-warning'],
        'CLASS': ['QSO', 'STAR', 'GALAXY'],
        'MASK': [0, 2, 1],
        'ZWARN': [0, 0, 1],
        **{
            magnitude: [float(index + 1), 10.0, 20.0]
            for index, magnitude in enumerate(MAGNITUDES)
        },
    })
    catalog.loc[0, 'MAG_GAAP_u'] = -99.0
    catalog_path = tmp_path / 'training_catalog.csv'
    catalog.to_csv(catalog_path)

    prepared_catalog = load_and_prepare_training_catalog(str(catalog_path))

    assert prepared_catalog['ID'].tolist() == ['kept']
    assert prepared_catalog['SPECTYPE'].tolist() == ['QSO']
    assert 'CLASS' not in prepared_catalog
    assert prepared_catalog.index.equals(pd.RangeIndex(1))
    assert pd.isna(prepared_catalog.loc[0, 'MAG_GAAP_u'])
    assert pd.isna(prepared_catalog.loc[0, 'u-g'])
    assert prepared_catalog.loc[0, 'g-r'] == -1.0
    assert prepared_catalog.loc[0, 'g/r'] == 2.0 / 3.0
