"""Unit tests for UncertaintyRNN using lightweight synthetic clinical sequences."""

import pytest
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models.uncertainty_rnn import VALID_UNCERTAINTY_TYPES, UncertaintyRNN


def _build_synthetic_dataset() -> SampleDataset:
    """Creates a minimal in-memory SampleDataset with mismatched sequence lengths."""
    samples = [
        {
            "patient_id": "p0",
            "visit_id": "v0",
            "conditions": [["401.9", "250.00"], ["428.0"]],
            "procedures": [["96.04"]],
            "mortality": 0,
        },
        {
            "patient_id": "p1",
            "visit_id": "v1",
            "conditions": [["410.71"], ["584.9", "518.81"], ["995.92"]],
            "procedures": [["38.93"], ["96.71"]],
            "mortality": 1,
        },
        {
            "patient_id": "p2",
            "visit_id": "v2",
            "conditions": [["486"]],
            "procedures": [["87.44"]],
            "mortality": 0,
        },
        {
            "patient_id": "p3",
            "visit_id": "v3",
            "conditions": [["038.9", "785.52"], ["427.5"]],
            "procedures": [["99.60"], ["38.93"]],
            "mortality": 1,
        },
    ]
    return SampleDataset(
        samples=samples,
        input_schema={"conditions": "nested_sequence", "procedures": "nested_sequence"},
        output_schema={"mortality": "binary"},
        dataset_name="synthetic_icu",
    )


@pytest.mark.parametrize("uncertainty_type", VALID_UNCERTAINTY_TYPES)
def test_uncertainty_rnn_forward_and_backward(uncertainty_type: str) -> None:
    """Verifies instantiation, forward output shapes, and gradients for all modes."""
    torch.manual_seed(42)
    dataset = _build_synthetic_dataset()
    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    model = UncertaintyRNN(
        dataset=dataset,
        uncertainty_type=uncertainty_type,
        embedding_dim=8,
        rnn_dim=16,
        hidden_dim=8,
        annealing_steps=10,
    )
    model.train()
    outputs = model(**batch)

    assert set(outputs.keys()) == {"loss", "y_prob", "y_true"}
    assert outputs["loss"].ndim == 0
    assert torch.isfinite(outputs["loss"])
    assert outputs["y_prob"].shape == (2,)
    assert outputs["y_true"].shape == (2,)
    assert torch.all((outputs["y_prob"] >= 0.0) & (outputs["y_prob"] <= 1.0))

    outputs["loss"].backward()
    has_grad = any(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.parameters()
    )
    assert has_grad


def test_uncertainty_rnn_mc_sampling_variance() -> None:
    """Verifies Bayesian variants produce distinct stochastic samples in eval mode."""
    torch.manual_seed(7)
    dataset = _build_synthetic_dataset()
    loader = get_dataloader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    model = UncertaintyRNN(
        dataset=dataset,
        uncertainty_type="fully_bayesian",
        embedding_dim=8,
        rnn_dim=16,
        hidden_dim=8,
    )
    model.eval()
    with torch.no_grad():
        sample_1 = model(**batch)["y_prob"]
        sample_2 = model(**batch)["y_prob"]

    assert not torch.allclose(sample_1, sample_2)


def test_uncertainty_rnn_invalid_type_raises() -> None:
    """Verifies invalid uncertainty_type raises a clear ValueError."""
    dataset = _build_synthetic_dataset()
    with pytest.raises(ValueError, match="Unsupported uncertainty_type"):
        UncertaintyRNN(dataset=dataset, uncertainty_type="invalid_mode")