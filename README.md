# KiDS-work

Machine-learning workflows for KiDS catalog classification, self-organizing
maps (SOMs), and photometric-redshift regression.

## Requirements

- Python >= 3.14
- [`uv`](https://docs.astral.sh/uv/)

Install the project and its dependencies with:

```bash
uv sync
```

## Data and models

Place the KiDS, DESI, and SDSS catalog files in `data/`. The training scripts
use:

- `data/KiDS_DR5_WCScut_x_DESI_DR1_columns.csv`
- `data/KiDS_DR5_WCScut_x_SDSS_DR17_columns.csv`
- `data/KiDS_DR5_WCScut_columns_15MLN.csv` (SOM training)

Trained models and preprocessors are stored in `saved_models/`.

## Workflows

Run scripts from the repository root:

```bash
uv run python scripts/train_classifier.py
uv run python scripts/train_photoz_regressor.py
uv run python scripts/train_som.py
```

Run classifier inference with:

```bash
uv run python scripts/inference_clf_kids.py
```

The notebooks in `notebooks/` contain exploratory analysis, training, and
visualization workflows. Run the tests with:

```bash
uv run pytest
```
