import numpy as np
import pandas as pd
import torch
import torch.distributions as distributions

from kids.inferencer import Inferencer


class DistributionModel(torch.nn.Module):
    def __init__(self, tuple_output: bool):
        super().__init__()
        self.tuple_output = tuple_output

    def forward(self, features):
        distribution = distributions.Normal(
            features[:, 0], torch.ones(features.shape[0], device=features.device)
        )
        return (distribution, torch.tensor(0.0)) if self.tuple_output else distribution


class IdentityPreprocessor:
    def transform(self, features):
        return features.to_numpy()


def test_infer_quasar_photoz_processes_all_cleaned_rows(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "MASK": [0, 2, 1],
            "feature": [1.0, 2.0, 3.0],
        }
    ).to_csv(input_path, index=False)

    inferencer = Inferencer(
        input_path=input_path,
        preprocessor=IdentityPreprocessor(),
        features=["feature"],
        chunksize=2,
    )
    inferencer._prepare_chunk = lambda chunk: chunk.loc[chunk["MASK"] <= 1]
    inferencer.infer_quasar_photoz(
        model=DistributionModel(tuple_output=False),
        output_path=output_path,
    )

    result = pd.read_csv(output_path)
    assert result["ID"].tolist() == [1, 3]
    np.testing.assert_allclose(result["PREDICTED_Z"], [1.0, 3.0])
    np.testing.assert_allclose(result["PREDICTED_Z_STD"], [1.0, 1.0])


def test_infer_quasar_photoz_supports_tuple_model_output(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame({"MASK": [0], "feature": [4.0]}).to_csv(input_path, index=False)

    inferencer = Inferencer(
        input_path=input_path,
        preprocessor=IdentityPreprocessor(),
        features=["feature"],
    )
    inferencer._prepare_chunk = lambda chunk: chunk.loc[chunk["MASK"] <= 1]
    inferencer.infer_quasar_photoz(
        model=DistributionModel(tuple_output=True),
        output_path=output_path,
    )

    result = pd.read_csv(output_path)
    np.testing.assert_allclose(result["PREDICTED_Z"], [4.0])
    np.testing.assert_allclose(result["PREDICTED_Z_STD"], [1.0])
