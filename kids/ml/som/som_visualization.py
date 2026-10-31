import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import load

from kids.catalog.training_catalog import (
    load_and_prepare_training_catalog,
    merge_training_catalogs,
)
from kids.features import clf_features
from kids.ml.classifier import XGBClf
from kids.ml.som import MySOM, SOMConfig
from kids.ml.training_data import TrainingData


@dataclass
class SOMVisualizationData:
    som: MySOM
    classifier: XGBClf
    catalog: pd.DataFrame
    raw_test: pd.DataFrame
    features: np.ndarray
    true_class_ids: np.ndarray
    bmus: np.ndarray
    bmu_distances: np.ndarray
    probabilities: np.ndarray
    predicted_class_ids: np.ndarray
    confidence: np.ndarray
    assignments: pd.DataFrame
    cell_stats: pd.DataFrame


def load_visualization_models(
    som_path: Path,
    preprocessor_path: Path,
    classifier_path: Path,
    som_config: SOMConfig,
    device: str | None = None,
) -> tuple[MySOM, object, XGBClf]:
    resolved_device = device or som_config.resolved_device()
    preprocessor = load(preprocessor_path)
    som = MySOM(**som_config.model_parameters(len(clf_features)))
    som.load_state_dict(
        torch.load(som_path, map_location=resolved_device, weights_only=True)
    )
    som.eval()

    classifier = XGBClf()
    classifier.load(classifier_path)
    expected_shape = (som_config.x, som_config.y, len(clf_features))
    if tuple(som.weights.shape) != expected_shape:
        raise ValueError(
            f"Expected SOM weights with shape {expected_shape}, "
            f"got {tuple(som.weights.shape)}"
        )
    return som, preprocessor, classifier


def load_test_data(
    desi_catalog_path: Path,
    sdss_catalog_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    desi = load_and_prepare_training_catalog(desi_catalog_path)
    sdss = load_and_prepare_training_catalog(sdss_catalog_path)
    catalog = merge_training_catalogs(desi, sdss)
    training_data = TrainingData(catalog, clf_features, train_size=0.6)
    features, true_class_ids = training_data.get_test_clf()
    raw_test = training_data.data.loc[training_data.idx_test].reset_index(drop=True)
    true_class_ids = np.asarray(true_class_ids, dtype=int)
    if len(raw_test) != len(features) or len(features) != len(true_class_ids):
        raise ValueError("Raw test rows, features, and labels have different lengths")
    return catalog, raw_test, np.asarray(features), true_class_ids


def assign_test_objects(
    som: MySOM,
    preprocessor: object,
    classifier: XGBClf,
    raw_test: pd.DataFrame,
    features: np.ndarray,
    true_class_ids: np.ndarray,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    som_features = preprocessor.transform(raw_test.loc[:, clf_features])
    resolved_device = device or str(som.device)
    with torch.no_grad():
        tensor = torch.as_tensor(som_features, dtype=torch.float32, device=resolved_device)
        bmus = som.identify_bmus(tensor).cpu().numpy()
        weights = som.weights[bmus[:, 0], bmus[:, 1]]
        bmu_distances = torch.linalg.vector_norm(tensor - weights, dim=1).cpu().numpy()

    probabilities = np.asarray(classifier.model.predict_proba(features))
    predicted_class_ids = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    if probabilities.shape[1] != len(TrainingData.CLASS_TO_ID):
        raise ValueError("Classifier output does not match the known class mapping")
    if not np.isclose(probabilities.sum(axis=1), 1).all():
        raise ValueError("Classifier probabilities do not sum to one")
    if len(true_class_ids) != len(bmus):
        raise ValueError("Test labels and SOM assignments have different lengths")
    return bmus, bmu_distances, probabilities, predicted_class_ids, confidence


def build_assignments(
    bmus: np.ndarray,
    bmu_distances: np.ndarray,
    true_class_ids: np.ndarray,
    predicted_class_ids: np.ndarray,
    confidence: np.ndarray,
    som_y: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": bmus[:, 0] * som_y + bmus[:, 1],
            "bmu_x": bmus[:, 0],
            "bmu_y": bmus[:, 1],
            "bmu_distance": bmu_distances,
            "true_class_id": true_class_ids,
            "predicted_class_id": predicted_class_ids,
            "confidence": confidence,
            "correct": predicted_class_ids == true_class_ids,
        }
    )


def build_cell_stats(
    assignments: pd.DataFrame,
    class_mapping: dict[int, str] | None = None,
) -> pd.DataFrame:
    class_mapping = class_mapping or TrainingData.ID_TO_CLASS
    stats = assignments.groupby("cell_id").agg(
        count=("correct", "size"),
        error_rate=("correct", lambda values: 1 - values.mean()),
        mean_confidence=("confidence", "mean"),
        accuracy=("correct", "mean"),
        mean_bmu_distance=("bmu_distance", "mean"),
        bmu_x=("bmu_x", "first"),
        bmu_y=("bmu_y", "first"),
    )
    stats["calibration_gap"] = stats["mean_confidence"] - stats["accuracy"]

    counts = pd.crosstab(assignments["cell_id"], assignments["true_class_id"])
    counts = counts.reindex(
        index=stats.index, columns=sorted(class_mapping), fill_value=0
    )
    fractions = counts.div(stats["count"], axis=0)
    for class_id, class_name in class_mapping.items():
        stats[f"true_fraction_{class_name.lower()}"] = fractions[class_id]

    stats["class_purity"] = fractions.max(axis=1)
    stats["class_entropy"] = -(
        fractions * np.log(fractions.replace(0, np.nan))
    ).sum(axis=1)
    for class_id, class_name in class_mapping.items():
        class_key = class_name.lower()
        true_positive = (
            (assignments["true_class_id"] == class_id)
            & (assignments["predicted_class_id"] == class_id)
        ).groupby(assignments["cell_id"]).sum()
        predicted_count = (assignments["predicted_class_id"] == class_id).groupby(
            assignments["cell_id"]
        ).sum()
        actual_count = (assignments["true_class_id"] == class_id).groupby(
            assignments["cell_id"]
        ).sum()
        stats[f"precision_{class_key}"] = (
            true_positive.reindex(stats.index, fill_value=0)
            / predicted_count.reindex(stats.index, fill_value=0).replace(0, np.nan)
        )
        stats[f"recall_{class_key}"] = (
            true_positive.reindex(stats.index, fill_value=0)
            / actual_count.reindex(stats.index, fill_value=0).replace(0, np.nan)
        )
    return stats


def prepare_visualization_data(
    desi_catalog_path: Path,
    sdss_catalog_path: Path,
    som_path: Path,
    preprocessor_path: Path,
    classifier_path: Path,
    som_config: SOMConfig,
    device: str | None = None,
) -> SOMVisualizationData:
    catalog, raw_test, features, true_class_ids = load_test_data(
        desi_catalog_path, sdss_catalog_path
    )
    som, preprocessor, classifier = load_visualization_models(
        som_path, preprocessor_path, classifier_path, som_config, device
    )
    bmus, distances, probabilities, predicted, confidence = assign_test_objects(
        som, preprocessor, classifier, raw_test, features, true_class_ids, device
    )
    assignments = build_assignments(
        bmus, distances, true_class_ids, predicted, confidence, som_config.y
    )
    return SOMVisualizationData(
        som=som,
        classifier=classifier,
        catalog=catalog,
        raw_test=raw_test,
        features=features,
        true_class_ids=true_class_ids,
        bmus=bmus,
        bmu_distances=distances,
        probabilities=probabilities,
        predicted_class_ids=predicted,
        confidence=confidence,
        assignments=assignments,
        cell_stats=build_cell_stats(assignments),
    )


def _figure_path(title: str, output_dir: Path) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return output_dir / f"{slug}.png"


def plot_cell_map(
    values: np.ndarray | pd.Series,
    title: str,
    *,
    cell_stats: pd.DataFrame,
    som_shape: tuple[int, int],
    output_dir: Path | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    values = np.asarray(values)
    n_x, n_y = som_shape
    if values.shape == som_shape:
        grid = values.copy()
    else:
        if values.size != len(cell_stats):
            raise ValueError(
                f"Expected one value per occupied cell ({len(cell_stats)}) "
                f"or a full SOM grid {som_shape}; got {values.shape}."
            )
        grid = np.full(som_shape, np.nan)
        grid[
            cell_stats["bmu_x"].to_numpy(dtype=int),
            cell_stats["bmu_y"].to_numpy(dtype=int),
        ] = values

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    if center is not None:
        limit = max(
            abs(np.nanmin(grid - center)),
            abs(np.nanmax(grid - center)),
        )
        vmin, vmax = center - limit, center + limit
    image = ax.imshow(
        grid.T,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set(title=title, xlabel="SOM x", ylabel="SOM y")
    fig.colorbar(image, ax=ax, label=title)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(_figure_path(title, output_dir), dpi=150)
    return fig, ax


def plot_som_component(
    som: MySOM,
    feature_name: str,
    *,
    cell_stats: pd.DataFrame,
    output_dir: Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    if feature_name not in clf_features:
        raise ValueError(f"{feature_name!r} is not a feature in the saved SOM")
    values = som.weights[:, :, clf_features.index(feature_name)].detach().cpu().numpy()
    return plot_cell_map(
        values,
        f"SOM component plane: {feature_name}",
        cell_stats=cell_stats,
        som_shape=(som.x, som.y),
        output_dir=output_dir,
        cmap="coolwarm",
    )


def plot_standard_maps(
    data: SOMVisualizationData,
    *,
    output_dir: Path | None = None,
) -> list[tuple[plt.Figure, plt.Axes]]:
    stats = data.cell_stats
    shape = (data.som.x, data.som.y)
    maps = [
        (stats["count"], "Cell population (test objects)", "viridis", None, None, None),
        (stats["error_rate"], "Error rate (fraction misclassified)", "magma", 0, 1, None),
        (stats["mean_confidence"], "Mean XGBoost confidence", "viridis", 0, 1, None),
        (stats["calibration_gap"], "Calibration gap (confidence - accuracy)", "coolwarm", None, None, 0),
        (stats["class_purity"], "Class purity (dominant true fraction)", "viridis", 0, 1, None),
        (stats["class_entropy"], "Class entropy", "viridis", 0, np.log(len(TrainingData.CLASS_TO_ID)), None),
    ]
    return [
        plot_cell_map(
            values,
            title,
            cell_stats=stats,
            som_shape=shape,
            output_dir=output_dir,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
        )
        for values, title, cmap, vmin, vmax, center in maps
    ]
