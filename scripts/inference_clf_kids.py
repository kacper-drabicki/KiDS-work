from pathlib import Path
from kids.inferencer import Inferencer
from kids.catalog.training_catalog import load_and_prepare_training_catalog, merge_training_catalogs
from kids.classifiers import XGBClf
from kids.training_data import TrainingData

DATA_DIR = Path("data")
DESI_CATALOG_PATH = DATA_DIR / "KiDS_DR5_WCScut_x_DESI_DR1_columns.csv"
SDSS_CATALOG_PATH = DATA_DIR / "KiDS_DR5_WCScut_x_SDSS_DR17_columns.csv"

INPUT_PATH = DATA_DIR / "KiDS_DR5_WCScut_columns.csv"
OUTPUT_PATH = DATA_DIR / "inference/inference_predictions.csv"

def main():

    print('Loading and preparing training catalogs...')
    training_desi = load_and_prepare_training_catalog(DESI_CATALOG_PATH)
    training_sdss = load_and_prepare_training_catalog(SDSS_CATALOG_PATH)

    training_catalog = merge_training_catalogs(training_desi, training_sdss)
    training_data = TrainingData(training_catalog)

    print('Loading model...')
    model = XGBClf()
    model.load("saved_models/classification/xgb_clf_2026-10-09T14-12-49Z.json")

    print('Running inference...')
    inference = Inferencer(
        input_path=INPUT_PATH,
        preprocessor=training_data.preprocess_features
    )

    inference.infer_class(
        model=model,
        output_path=OUTPUT_PATH
    )
    print(f'Inference completed. Predictions saved to {OUTPUT_PATH}')

if __name__ == "__main__":
    main()
