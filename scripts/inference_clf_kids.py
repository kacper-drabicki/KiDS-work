from kids.catalog.inference_catalog import InferenceCatalog
from kids.catalog.training_catalog import (
    load_and_prepare_training_catalog,
    merge_training_catalogs,
)
from kids.classifiers import XGBClf
from kids.training_data import TrainingData

training_desi = load_and_prepare_training_catalog(
    "data/KiDS_DR5_WCScut_x_DESI_DR1_columns.csv"
)
training_sdss = load_and_prepare_training_catalog(
    "data/KiDS_DR5_WCScut_x_SDSS_DR17_columns.csv"
)

training_catalog = merge_training_catalogs(training_desi, training_sdss)
training_data = TrainingData(training_catalog)

model = XGBClf()
model.load("saved_models/classification/xgb_clf_2026-10-09T14-12-49Z.json")

inference = InferenceCatalog(
    model=model,
    preprocessor=training_data.preprocess_features,
    chunksize=100_000,
)

inference.predict_csv(
    input_path="data/inference_catalog.csv",
    output_path="data/inference_predictions.csv",
)
