import datetime
from pathlib import Path

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from kids.logger import logger
from kids.catalog.preparation import load_and_prepare_training_catalog, merge_training_catalogs
from kids.catalog.features import PHOTOZ_FEATURES
from kids.ml.training_data import TrainingData
from kids.ml.photoz.train_core import get_data_loader
from kids.ml.photoz.lightning_mdn import MDNLightningModule
from configs.regressors_config import MDNConfig
from configs.train_mdn_config import MDNTrainingConfig

DATA_DIR = Path('data')
DESI_CATALOG_PATH = DATA_DIR / 'KiDS_DR5_WCScut_x_DESI_DR1_columns.csv'
SDSS_CATALOG_PATH = DATA_DIR / 'KiDS_DR5_WCScut_x_SDSS_DR17_columns.csv'

timestamp = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%SZ')
MODEL_DIR = Path('saved_models/photoz')
MODEL_PATH = MODEL_DIR / f'mdn_photoz_{timestamp}.pt'
CHECKPOINT_DIR = MODEL_DIR / 'lightning_checkpoints'

def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(42, workers=True)

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

    train_loader = get_data_loader(
        X_train, y_train, batch_size=training_config.batch_size
    )
    val_loader = get_data_loader(
        X_val, y_val, batch_size=training_config.batch_size
    )
    model = MDNLightningModule(MDNConfig(), training_config)
    checkpoint = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename='mdn-{epoch:03d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
    )
    trainer = Trainer(
        accelerator='auto',
        devices='auto',
        max_epochs=training_config.n_epochs,
        callbacks=[
            checkpoint,
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=training_config.early_stopping_patience,
                min_delta=training_config.min_delta,
            ),
            LearningRateMonitor(logging_interval='step'),
        ],
        logger=CSVLogger(save_dir=str(MODEL_DIR), name='mdn_logs'),
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    logger.info('Training starts ...')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_model = MDNLightningModule.load_from_checkpoint(
        checkpoint.best_model_path,
        model_config=MDNConfig(),
        training_config=training_config,
    )
    best_model.model.save(MODEL_PATH)
    logger.info(f'Model saved as {MODEL_PATH}')

if __name__ == '__main__':
    main()
