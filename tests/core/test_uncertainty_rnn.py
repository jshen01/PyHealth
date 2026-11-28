import inspect
import pytest
import torch

from pyhealth.datasets import get_dataloader
from pyhealth.datasets.sample_dataset import create_sample_dataset
from pyhealth.models import UncertaintyRNN


def _build_synthetic_dataset(tmp_path):
    """Create a tiny synthetic SampleDataset using PyHealth's create_sample_dataset."""
    raw_samples = [
        {
            "visit_id": "v0",
            "patient_id": "p0",
            "conditions": ["401.9", "250.00", "428.0"],
            "procedures": ["96.04", "38.93"],
            "mortality": 0,
        },
        {
            "visit_id": "v1",
            "patient_id": "p1",
            "conditions": ["584.9", "518.81", "995.92"],
            "procedures": ["39.95", "96.71"],
            "mortality": 1,
        },
        {
            "visit_id": "v2",
            "patient_id": "p2",
            "conditions": ["250.00", "401.9", "414.01"],
            "procedures": ["38.93", "96.04"],
            "mortality": 0,
        },
        {
            "visit_id": "v3",
            "patient_id": "p3",
            "conditions": ["995.92", "584.9", "518.81"],
            "procedures": ["96.71", "39.95"],
            "mortality": 1,
        },
        {
            "visit_id": "v4",
            "patient_id": "p4",
            "conditions": ["414.01", "401.9"],
            "procedures": ["38.93", "38.93"],
            "mortality": 0,
        },
        {
            "visit_id": "v5",
            "patient_id": "p5",
            "conditions": ["428.0", "518.81", "995.92"],
            "procedures": ["96.04", "96.71"],
            "mortality": 1,
        },
    ]

    input_schema = {"conditions": "sequence", "procedures": "sequence"}
    output_schema = {"mortality": "binary"}

    sig = inspect.signature(create_sample_dataset)
    kwargs = {
        "samples": raw_samples,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "dataset_name": "synthetic_mortality",
        "task_name": "mortality_prediction",
    }
    if "in_memory" in sig.parameters:
        kwargs["in_memory"] = True
    elif "cache_dir" in sig.parameters:
        kwargs["cache_dir"] = str(tmp_path)
    elif "path" in sig.parameters:
        kwargs["path"] = str(tmp_path)

    return create_sample_dataset(**kwargs)


@pytest.mark.parametrize(
    "uncertainty_type",
    [
        "deterministic",
        "bayesian_embeddings",
        "bayesian_output",
        "bayesian_hidden_output",
        "fully_bayesian",
    ],
)
def test_uncertainty_rnn_forward_and_backward(tmp_path, uncertainty_type):
    """Verify forward pass, ELBO/NLL loss, and gradient flow for all 5 regimes."""
    torch.manual_seed(42)
    dataset = _build_synthetic_dataset(tmp_path)
    loader = get_dataloader(dataset, batch_size=3, shuffle=False)
    batch = next(iter(loader))

    model = UncertaintyRNN(
        dataset=dataset,
        uncertainty_type=uncertainty_type,
        embedding_dim=8,
        rnn_dim=16,
        hidden_dim=8,
        annealing_steps=5,
        num_train_samples=len(dataset),
    )

    model.train()
    output = model(**batch)

    assert "loss" in output
    assert "y_prob" in output
    assert "y_true" in output
    assert torch.isfinite(output["loss"])
    assert output["y_prob"].shape == (3,)
    assert torch.all((output["y_prob"] >= 0.0) & (output["y_prob"] <= 1.0))

    output["loss"].backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )
    assert has_grad


def test_uncertainty_rnn_mc_sampling_variance(tmp_path):
    """Verify Monte Carlo evaluation produces non-zero epistemic variance."""
    torch.manual_seed(42)
    dataset = _build_synthetic_dataset(tmp_path)
    loader = get_dataloader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    model = UncertaintyRNN(
        dataset=dataset,
        uncertainty_type="fully_bayesian",
        embedding_dim=8,
        rnn_dim=16,
        hidden_dim=8,
        num_train_samples=len(dataset),
    )

    model.eval()
    with torch.no_grad():
        passes = torch.stack([model(**batch)["y_prob"] for _ in range(10)], dim=0)

    assert passes.shape == (10, 4)
    assert passes.std(dim=0).mean().item() > 0.0


def test_uncertainty_rnn_invalid_type_raises(tmp_path):
    """Verify invalid uncertainty_type raises ValueError."""
    dataset = _build_synthetic_dataset(tmp_path)
    with pytest.raises(ValueError):
        UncertaintyRNN(
            dataset=dataset,
            uncertainty_type="invalid_mode",
        )