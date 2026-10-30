from pathlib import Path

from kids.logger import logger
from kids.ml.som import SOMConfig
from kids.ml.som_training import train_som

DATA_DIR = Path("data")
MODEL_DIR = Path("saved_models/som")
DATA_PATH = DATA_DIR / "KiDS_DR5_WCScut_columns_15MLN.csv"
PREDICTIONS_PATH = DATA_DIR / "inference/KiDS_DR5_WCScut_columns_SPECTYPE.csv"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor_robust_scaler.joblib"
MODEL_PATH = MODEL_DIR / "test_som_60_60.pth"


def main() -> None:
    config = SOMConfig()
    logger.info("Loading SOM catalog and fitting preprocessor ...")
    result = train_som(
        config,
        data_path=DATA_PATH,
        predictions_path=PREDICTIONS_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        model_path=MODEL_PATH,
    )
    logger.info("SOM training completed.")
    logger.info("Final quantization error: %.6f", result.quantization_errors[-1])
    logger.info("Final topographic error: %.6f", result.topographic_errors[-1])
    logger.info("Preprocessor saved to %s", PREPROCESSOR_PATH)
    logger.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
