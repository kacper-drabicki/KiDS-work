import torch
from pathlib import Path
from kids.catalog.features import PHOTOZ_FEATURES
from kids.catalog.preparation import (
    load_and_prepare_training_catalog,
    merge_training_catalogs,
)
from kids.inferencer import Inferencer
from kids.logger import logger
from kids.ml.photoz.regressors import (
    BayesianNet,
    MixtureDensityNetwork,
)
from kids.ml.training_data import TrainingData
from configs.regressors_config import BNConfig, MDNConfig


DATA_DIR = Path("data")
DESI_CATALOG_PATH = DATA_DIR / "KiDS_DR5_WCScut_x_DESI_DR1_columns.csv"
SDSS_CATALOG_PATH = DATA_DIR / "KiDS_DR5_WCScut_x_SDSS_DR17_columns.csv"

INPUT_PATH = DATA_DIR / "KiDS_DR5_WCScut_columns.csv"
OUTPUT_PATH = DATA_DIR / "inference/KiDS_DR5_WCScut_columns_PHOTOZ.csv"
MODEL_PATH = Path("saved_models/photoz/mdn_photoz_2026-10-31T11-07-15Z.pt")
# Available model types: "bnn" (Bayesian Neural Network) or "mdn" (Mixture Density Network)
MODEL_TYPE = "mdn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BNN_SAMPLES = 100


def main() -> None:
    logger.info("Loading and preparing training catalogs...")
    training_desi = load_and_prepare_training_catalog(DESI_CATALOG_PATH)
    training_sdss = load_and_prepare_training_catalog(SDSS_CATALOG_PATH)
    training_catalog = merge_training_catalogs(training_desi, training_sdss)
    training_data = TrainingData(training_catalog, features=PHOTOZ_FEATURES)

    if MODEL_TYPE == "bnn":
        model = BayesianNet(BNConfig())
    else:
        model = MixtureDensityNetwork(MDNConfig())
    model.load_weights(str(MODEL_PATH), DEVICE)

    logger.info(f"Starting inference on {INPUT_PATH} using model {MODEL_PATH}...")
    inference = Inferencer(
        input_path=INPUT_PATH,
        preprocessor=training_data.preprocess_features,
        features=PHOTOZ_FEATURES)
    inference.infer_quasar_photoz(
        model=model,
        output_path=OUTPUT_PATH,
        device=DEVICE,
        n_samples=BNN_SAMPLES if MODEL_TYPE == "bnn" else 1)
    logger.info(f"Inference completed. Predictions saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
