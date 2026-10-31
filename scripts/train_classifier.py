import datetime
from pathlib import Path

from kids.logger import logger
from kids.catalog.preparation import load_and_prepare_training_catalog, merge_training_catalogs
from kids.ml.training_data import TrainingData
from kids.ml.classification.xgbclf import XGBClf, XGBConfig
from kids.catalog.features import CLF_FEATURES

DATA_DIR = Path('data')
DESI_CATALOG_PATH = DATA_DIR / 'KiDS_DR5_WCScut_x_DESI_DR1_columns.csv'
SDSS_CATALOG_PATH = DATA_DIR / 'KiDS_DR5_WCScut_x_SDSS_DR17_columns.csv'

timestamp = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%SZ')
MODEL_DIR = Path('saved_models/classification')
MODEL_PATH = MODEL_DIR / f'xgb_clf_{timestamp}.json'

def main() -> None:

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info('Loading and preparing training catalogs ...')
    kids_x_desi = load_and_prepare_training_catalog(DESI_CATALOG_PATH)
    kids_x_sdss = load_and_prepare_training_catalog(SDSS_CATALOG_PATH)
    logger.info('Merging training catalogs ...')
    merged_catalogs = merge_training_catalogs(kids_x_desi, kids_x_sdss)

    logger.info('Preparing training data ...')
    training_data = TrainingData(merged_catalogs, features=CLF_FEATURES, train_size=0.6)
    X_train, y_train = training_data.get_train_clf()
    X_val, y_val = training_data.get_val_clf()

    xgb = XGBClf(XGBConfig())

    logger.info('Training starts ...')
    xgb.train(X_train, y_train, X_val, y_val)

    xgb.save(MODEL_PATH)
    logger.info(f'Model saved as {MODEL_PATH}')


if __name__ == '__main__':
    main()
