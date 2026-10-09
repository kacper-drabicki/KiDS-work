from load_kids_catalog import load_and_process, concat_two_catalogs
from data_exp import ExperimentalData
from models import XGBClf, XGBConfig
from sklearn.metrics import classification_report
import datetime

timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%SZ")
MODEL_PATH = f'saved_models/classification/xgb_clf_{timestamp}.json'

def main():

    kids_x_desi = load_and_process('data/KiDS_DR5_WCScut_x_DESI_DR1_columns.csv')
    kids_x_sdss = load_and_process('data/KiDS_DR5_WCScut_x_SDSS_DR17_columns.csv')
    kids_x_desi_and_sdss = concat_two_catalogs(kids_x_desi, kids_x_sdss)

    data_exp = ExperimentalData(kids_x_desi_and_sdss)
    X_train, y_train = data_exp.get_train_clf()
    X_val, y_val = data_exp.get_val_clf()

    xgb = XGBClf(XGBConfig())

    print("Training starts ...")
    xgb.train(X_train, y_train, X_val, y_val)

    xgb.save(MODEL_PATH)
    print(f'Model saved as {MODEL_PATH}')


if __name__ == "__main__":
    main()
