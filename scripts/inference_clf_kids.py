from pathlib import Path
from kids.inferencer import Inferencer
from kids.catalog.training_catalog import load_and_prepare_training_catalog, merge_training_catalogs
from kids.classifiers import XGBClf
from kids.training_data import TrainingData

def main():

    desi_path = Path("data/KiDS_DR5_WCScut_x_DESI_DR1_columns.csv")
    sdss_path = Path("data/KiDS_DR5_WCScut_x_SDSS_DR17_columns.csv")

    input_path = Path("data/KiDS_DR5_WCScut_columns.csv")
    output_path = Path("data/inference/inference_predictions.csv")

    print('Loading and preparing training catalogs...')
    training_desi = load_and_prepare_training_catalog(desi_path)
    training_sdss = load_and_prepare_training_catalog(sdss_path)

    training_catalog = merge_training_catalogs(training_desi, training_sdss)
    training_data = TrainingData(training_catalog)

    print('Loading model...')
    model = XGBClf()
    model.load("saved_models/classification/xgb_clf_2026-10-09T14-12-49Z.json")

    print('Running inference...')
    inference = Inferencer(
        input_path=input_path,
        preprocessor=training_data.preprocess_features
    )

    inference.infer_class(
        model=model,
        output_path=output_path
    )
    print(f'Inference completed. Predictions saved to {output_path}')

if __name__ == "__main__":
    main()
