import datetime
from pathlib import Path
from kids.catalog.training_catalog import load_and_prepare_training_catalog, merge_training_catalogs
from kids.training_data import TrainginData
from kids.classifiers import XGBClf, XGBConfig

DATA_DIR = Path("data")
DESI_CATALOG_PATH = DATA_DIR / "KiDS_DR5_WCScut_x_DESI_DR1_columns.csv"
SDSS_CATALOG_PATH = DATA_DIR / "KiDS_DR5_WCScut_x_SDSS_DR17_columns.csv"

timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%SZ")
MODEL_DIR = Path("saved_models/classification")
MODEL_PATH = MODEL_DIR / f"xgb_clf_{timestamp}.json"

def main():

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    kids_x_desi = load_and_prepare_training_catalog(DESI_CATALOG_PATH)
    kids_x_sdss = load_and_prepare_training_catalog(SDSS_CATALOG_PATH)
    merged_catalogs = merge_training_catalogs(kids_x_desi, kids_x_sdss)

    training_data = TrainginData(merged_catalogs)
    X_train, y_train = training_data.get_train_clf()
    X_val, y_val = training_data.get_val_clf()

    xgb = XGBClf(XGBConfig())

    print("Training starts ...")
    xgb.train(X_train, y_train, X_val, y_val)

    xgb.save(MODEL_PATH)
    print(f'Model saved as {MODEL_PATH}')


if __name__ == "__main__":
    main()
