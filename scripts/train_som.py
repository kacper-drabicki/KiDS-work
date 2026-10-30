import datetime
import torch
from pathlib import Path
from joblib import dump
from dataclasses import asdict
from kids.logger import logger
from kids.catalog.preparation import load_and_prepare_inference_catalog
from kids.ml.som.som_utils import get_preprocessor
from kids.catalog.features import CLF_FEATURES
from kids.ml.som.som import SOMConfig, MySOM


DATA_DIR = Path("data")
DATA_PATH = DATA_DIR / "KiDS_DR5_WCScut_columns_15MLN.csv"

timestamp = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%SZ')
MODEL_DIR = Path("saved_models/som")
MODEL_PATH = MODEL_DIR / f"som_{timestamp}.pth"
PREPROCESSOR_PATH = MODEL_DIR / f"preprocessor_{timestamp}.joblib"

N_SAMPLES_FOR_PREPROCESSOR = 2_500_000

def main() -> None:
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and preparing catalog ...")
    df = load_and_prepare_inference_catalog(DATA_PATH)

    logger.info("Fitting preprocessor ...")
    preprocessor = get_preprocessor(df.loc[:, CLF_FEATURES], N_SAMPLES_FOR_PREPROCESSOR)
    dump(preprocessor, PREPROCESSOR_PATH)
    logger.info(f"Preprocessor saved as {PREPROCESSOR_PATH}")

    logger.info("Transforming features ...")
    transformed_features = torch.tensor(preprocessor.transform(df.loc[:, CLF_FEATURES]), dtype=torch.float32)

    config = asdict(SOMConfig())

    som = MySOM(**config)
    
    logger.info("Training SOM ...")
    som.initialize_weights(data=transformed_features, mode=som.initialization_mode)
    q_errors, t_errors = som.fit(data=transformed_features)
    logger.info(f"Training completed. Quantization error: {q_errors[-1]}, Topographic error: {t_errors[-1]}")

    torch.save(som.state_dict(), MODEL_PATH)
    logger.info(f"SOM model saved as {MODEL_PATH}")
    

if __name__ == "__main__":
    main()
