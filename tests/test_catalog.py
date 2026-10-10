import pandas as pd
import pytest

from kids.catalog.catalog import add_colors, clean_mask, replace_sentinel
from kids.catalog.training_catalog import clean_zwarn, merge_training_catalogs, rename_class
from kids.features import COLORS, MAGNITUDES


def test_clean_mask_keeps_only_acceptable_rows():
    data = pd.DataFrame({"MASK": [0, 1, 2, -1]})
    result = clean_mask(data)
    assert result.index.tolist() == [0, 1, 3]


@pytest.mark.parametrize("column", ["ZWARN", "ZWARNING"])
def test_clean_zwarn_supports_both_catalog_column_names(column):
    data = pd.DataFrame({column: [0, 1, 0]})
    result = clean_zwarn(data)
    assert result.index.tolist() == [0, 2]


def test_replace_sentinel_replaces_both_values():
    data = pd.DataFrame({"value": [-99.0, 99.0, 1.5]})
    result = replace_sentinel(data)
    assert result.loc[[0, 1], "value"].isna().all()
    assert result.loc[2, "value"] == 1.5


def test_add_colors_creates_expected_values_and_columns():
    data = pd.DataFrame({name: [float(index)] for index, name in enumerate(MAGNITUDES)})
    result = add_colors(data)
    assert result.shape[1] == len(MAGNITUDES) + len(COLORS)
    assert result.loc[0, "u-g"] == -1.0
    assert result.loc[0, "H-Ks"] == -1.0
    assert set(COLORS) == set(result.columns) - set(MAGNITUDES)


def test_training_catalog_helpers_preserve_expected_class_contract():
    data = pd.DataFrame({"CLASS": ["STAR", "GALAXY"]})
    result = rename_class(data)
    assert result.columns.tolist() == ["SPECTYPE"]


def test_merge_keeps_first_duplicate_and_resets_index():
    first = pd.DataFrame({"ID": [1, 2], "value": ["first", "only"]}, index=[10, 11])
    second = pd.DataFrame({"ID": [2, 3], "value": ["second", "new"]}, index=[20, 21])
    result = merge_training_catalogs(first, second)
    assert result["ID"].tolist() == [1, 2, 3]
    assert result["value"].tolist() == ["first", "only", "new"]
    assert result.index.tolist() == [0, 1, 2]
