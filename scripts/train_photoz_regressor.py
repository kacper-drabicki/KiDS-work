import datetime
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path

from kids.catalog.training_catalog import load_and_prepare_training_catalog, merge_training_catalogs
from kids.training_data import TrainingData
from kids.training_photoz_utils import get_data_loader, train_model
from kids.photoz_regressors import MixtureDensityNetwork, MDNConfig

DATA_DIR = Path('data')
DESI_CATALOG_PATH = DATA_DIR / 'KiDS_DR5_WCScut_x_DESI_DR1_columns.csv'
SDSS_CATALOG_PATH = DATA_DIR / 'KiDS_DR5_WCScut_x_SDSS_DR17_columns.csv'

timestamp = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%SZ')
MODEL_DIR = Path('saved_models/photoz')
MODEL_PATH = MODEL_DIR / f'mdn_photoz_{timestamp}.pt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256
LEARNING_RATE = 3E-4
N_EPOCHS = 200
PATIENCE = 30

def main():
    kids_x_desi = load_and_prepare_training_catalog(DESI_CATALOG_PATH)
    kids_x_sdss = load_and_prepare_training_catalog(SDSS_CATALOG_PATH)
    merged_catalogs = merge_training_catalogs(kids_x_desi, kids_x_sdss)

    training_data = TrainingData(merged_catalogs)
    X_train, y_train = training_data.get_train_photoz()
    X_val, y_val = training_data.get_val_photoz()

    train_loader = get_data_loader(X_train, y_train, batch_size=BATCH_SIZE)
    val_loader = get_data_loader(X_val, y_val, batch_size=BATCH_SIZE)
    
    model = MixtureDensityNetwork(MDNConfig())
    model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=N_EPOCHS)

    history = train_model(model, train_loader, val_loader, optimizer, DEVICE, n_epochs=N_EPOCHS, patience=PATIENCE, scheduler=scheduler)

    model.save(MODEL_PATH)

if __name__ == '__main__':
    main()


