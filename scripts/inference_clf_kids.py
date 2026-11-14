from pathlib import Path
from kids.logger import logger
from kids.inferencer import Inferencer
from kids.catalog.preparation import (
    load_and_prepare_training_catalog, merge_training_catalogs
)
from kids.ml.classification.xgbclf import XGBClf
from kids.ml.training_data import TrainingData
from kids.catalog.features import CLF_FEATURES

DATA_DIR = Path('data')
DESI_CATALOG_PATH = DATA_DIR / 'raw_data/KiDS_DR5_WCScut_x_DESI_DR1_columns.csv'
SDSS_CATALOG_PATH = DATA_DIR / 'raw_data/KiDS_DR5_WCScut_x_SDSS_DR17_columns.csv'

INPUT_PATH = DATA_DIR / 'raw_data/KiDS_DR5_WCScut_columns.csv'
OUTPUT_PATH = DATA_DIR / 'classification/KiDS_DR5_WCScut_columns_SPECTYPE_train_60.csv'

MODEL_PATH = Path('saved_models/classification/xgb_clf_2026-10-30T11-32-05Z.json')

def main() -> None:

    logger.info('Loading and preparing training catalogs...')
    training_desi = load_and_prepare_training_catalog(DESI_CATALOG_PATH)
    training_sdss = load_and_prepare_training_catalog(SDSS_CATALOG_PATH)

    training_catalog = merge_training_catalogs(training_desi, training_sdss)
    training_data = TrainingData(training_catalog, features=CLF_FEATURES, train_size=0.6)

    logger.info('Loading model...')
    model = XGBClf()
    model.load(MODEL_PATH)

    logger.info('Running inference...')
    inference = Inferencer(
        input_path=INPUT_PATH,
        preprocessor=training_data.preprocess_features,
        features=CLF_FEATURES)

    inference.infer_class(
        model=model,
        output_path=OUTPUT_PATH)
    logger.info(f'Inference completed. Predictions saved to {OUTPUT_PATH}')

if __name__ == "__main__":
    main()
