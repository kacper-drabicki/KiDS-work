# KiDS-work Project Guidelines


## Python and Dependencies

- Support Python `>=3.14`, as declared in `pyproject.toml`.
- Use `uv` for environment setup, dependency synchronization, and command execution.
- Add dependencies with `uv add`, not by editing `pyproject.toml` or `uv.lock` manually.
- Keep `uv.lock` synchronized after dependency changes.
- Before adding a dependency, check whether the standard library or an existing project dependency is sufficient. Prefer the libraries already used by this project, including NumPy, pandas, Polars, scikit-learn, XGBoost, PyTorch, Astropy, and Matplotlib.
- Preserve the existing Hatch build configuration and `kids` package layout.

## Project Architecture

- Keep reusable production code in `kids/`.
- Keep executable workflows in `scripts/`.
- Keep automated tests in `tests/`.
- Treat notebooks as exploratory or demonstrative unless explicitly asked to productionize their logic.
- Reuse existing modules and utilities, especially the catalog, training-data, feature, classifier, photo-z regressor, and inference modules. Avoid duplicating data preparation, feature definitions, model configuration, or inference logic.

## Scientific Data and Reproducibility

- Do not modify source catalog files or generated datasets unless explicitly requested.
- Do not commit large generated data files, trained model artifacts, credentials, or local environment files.
- Preserve the distinction between raw/catalog data in `data/`, trained model artifacts in `saved_models/`, reusable code in `kids/`, and exploratory analysis in `notebooks/`.
- Do not silently change feature lists, class labels, data splits, preprocessing, or model hyperparameters.
- Preserve random seeds and deterministic split behavior unless a change explicitly targets reproducibility.
- Changes affecting scientific results should state their expected effect and include a regression test or reproducible validation where practical.

## Code Style and Design

- Follow the existing Python style, naming patterns, and module boundaries.
- Prefer clear type annotations for public functions and methods.
- Prefer explicit, readable data transformations over compressed one-liners.
- Preserve public APIs unless a breaking change is explicitly requested.
- Avoid broad exception handling and silent fallback behavior.
- Keep comments limited to non-obvious scientific or implementation decisions.
- Make focused changes and avoid unrelated cleanup.
- Do not embed machine-specific paths.

## Models and Artifacts

- Preserve the existing model directories and naming conventions.
- Do not overwrite saved models unintentionally.
- When changing model serialization or loading behavior, test compatibility with existing JSON and PyTorch artifacts where applicable.
- Keep training and inference workflows explicit about input and output paths.

## Testing and Validation

- Run the smallest relevant validation first. The standard test command is:

  ```bash
  uv run pytest
  ```

- Add or update tests under `tests/` for behavioral changes.
- Preserve coverage of catalog loading, training-data splitting, and reproducibility behavior.
- For model-training or inference changes, validate normal execution and output shape/type expectations.
- Do not treat notebook execution as a replacement for automated tests.
- Follow the existing CI workflow: dependencies are installed with `uv sync --locked` and tests are run with `uv run pytest`.

## Documentation

- Update `README.md` when setup commands, supported Python versions, workflows, or input/output files change.
- Document new public modules, scripts, or workflows.
- Keep documentation concise and consistent with the existing README.



