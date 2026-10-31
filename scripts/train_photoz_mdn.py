import datetime
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path

from kids.logger import logger
from kids.catalog.preparation import load_and_prepare_training_catalog, merge_training_catalogs
from kids.catalog.features import PHOTOZ_FEATURES
from kids.ml.training_data import TrainingData
from kids.ml.photoz.train_core import get_data_loader, plot_training_history
from kids.ml.photoz.train_mdn import train_mdn, MDNTrainingConfig
from kids.ml.photoz.regressors import MixtureDensityNetwork, MDNConfig

DATA_DIR = Path('data')
DESI_CATALOG_PATH = DATA_DIR / 'KiDS_DR5_WCScut_x_DESI_DR1_columns.csv'
SDSS_CATALOG_PATH = DATA_DIR / 'KiDS_DR5_WCScut_x_SDSS_DR17_columns.csv'

timestamp = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%SZ')
MODEL_DIR = Path('saved_models/photoz')
MODEL_PATH = MODEL_DIR / f'mdn_photoz_{timestamp}.pt'

def main() -> None:
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info('Loading and preparing training catalogs ...')
    kids_x_desi = load_and_prepare_training_catalog(DESI_CATALOG_PATH)
    kids_x_sdss = load_and_prepare_training_catalog(SDSS_CATALOG_PATH)
    logger.info('Merging training catalogs ...')
    merged_catalogs = merge_training_catalogs(kids_x_desi, kids_x_sdss)

    logger.info('Preparing training data ...')
    training_data = TrainingData(merged_catalogs, features=PHOTOZ_FEATURES, train_size=0.8)
    X_train, y_train = training_data.get_train_photoz()
    X_val, y_val = training_data.get_val_photoz()

    training_config = MDNTrainingConfig()

    train_loader = get_data_loader(X_train, y_train, batch_size=training_config.batch_size)
    val_loader = get_data_loader(X_val, y_val, batch_size=training_config.batch_size)
    
    model = MixtureDensityNetwork(MDNConfig())
    model.to(training_config.device)

    optimizer = Adam(model.parameters(), lr=training_config.learning_rate)

    scheduler = OneCycleLR(optimizer=optimizer, 
                           steps_per_epoch=len(train_loader),
                           max_lr=training_config.max_lr, 
                           epochs=training_config.n_epochs)

    logger.info('Training starts ...')
    history = train_mdn(model=model, 
                        train_loader=train_loader, 
                        val_loader=val_loader, 
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=training_config)

    model.save(MODEL_PATH)
    logger.info(f'Model saved as {MODEL_PATH}')

    plot_training_history(history)

if __name__ == '__main__':
    main()


