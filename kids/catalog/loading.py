import pandas as pd

def load_catalog(file_path: str, index_col: int | None = None) -> pd.DataFrame:
    return pd.read_csv(file_path, index_col=index_col)
