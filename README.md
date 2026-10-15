# KiDS-work

Tools for training galaxy classification and photometric-redshift models from
KiDS catalog data.

## Requirements

- Python >= 3.14
- [`uv`](https://docs.astral.sh/uv/)

Install the project and its dependencies with:

```bash
uv sync
```

## Data

Place the required KiDS, DESI, and SDSS catalog files in `data/`. The training
scripts expect:

- `data/KiDS_DR5_WCScut_x_DESI_DR1_columns.csv`
- `data/KiDS_DR5_WCScut_x_SDSS_DR17_columns.csv`

## Usage

Run the training scripts from the repository root:

```bash
uv run python scripts/train_classifier.py
uv run python scripts/train_photoz_regressor.py
```

Models are saved under `saved_models/classification/` and
`saved_models/photoz/`.

To run the included classifier inference workflow:

```bash
uv run python scripts/inference_clf_kids.py
```

It reads `data/KiDS_DR5_WCScut_columns.csv` and writes predictions to
`data/inference/inference_predictions.csv`.

Run the test suite with:

```bash
uv run pytest
```

Jupyter notebooks with exploratory analyses are available in `notebooks/`.
