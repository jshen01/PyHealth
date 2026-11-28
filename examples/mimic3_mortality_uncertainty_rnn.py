"""Example & Ablation Study: Uncertainty Quantification for EHR Mortality Prediction.

Reproduces Dusenberry et al. (CHIL '20), "Analyzing the Role of Model Uncertainty
for Electronic Health Records," using PyHealth's ``UncertaintyRNN`` across five
uncertainty regimes:
  1. ``deterministic`` (deep ensemble across M random seeds)
  2. ``bayesian_embeddings``
  3. ``bayesian_output``
  4. ``bayesian_hidden_output``
  5. ``fully_bayesian``

Hardware & Benchmark Summary (12 GB Consumer GPU):
    Full-width paper hyperparameters (``rnn_dim=1024``, ``hidden_dim=512``,
    ``batch_size=256``) exceed 12 GB VRAM on multi-sequence ICU cohorts. All
    clinical experiments below were executed with ``embedding_dim=32``,
    ``rnn_dim=256``, ``hidden_dim=128``, and ``batch_size=32``:

    1. 3-Epoch Baseline Sweep (Unscaled KL Annealing, M=5 det / M=10 Bayes):
        Dataset   | Mode                   | AUC-ROC | AUC-PR | ECE    | Mean Spread (Max)
        ----------|------------------------|---------|--------|--------|------------------
        eICU      | deterministic (M=5)    | 0.6058  | 0.1178 | 0.0112 | 0.0517 (0.5430)
        eICU      | bayesian_output        | 0.5741  | 0.1144 | 0.0291 | 0.0375 (0.1455)
        eICU      | bayesian_hidden_output | 0.5777  | 0.1153 | 0.0164 | 0.0344 (0.1636)
        eICU      | bayesian_embeddings    | 0.5769  | 0.1134 | 0.0187 | 0.0826 (0.4773)
        eICU      | fully_bayesian         | 0.5857  | 0.1118 | 0.0138 | 0.1120 (0.5841)
        MIMIC-III | deterministic (M=5)    | 0.5452  | 0.1307 | 0.0111 | 0.0560 (0.2615)
        MIMIC-III | bayesian_output        | 0.5116  | 0.1220 | 0.0352 | 0.0352 (0.0720)
        MIMIC-III | bayesian_hidden_output | 0.5123  | 0.1235 | 0.0429 | 0.0380 (0.0930)
        MIMIC-III | bayesian_embeddings    | 0.5235  | 0.1368 | 0.0508 | 0.0395 (0.3747)
        MIMIC-III | fully_bayesian         | 0.5577  | 0.1435 | 0.0593 | 0.1064 (0.4082)

    2. 10-Epoch Scaled KL-Annealing & Tuned-LR Ablation (lr=1e-4 det / lr=5e-4 Bayes):
        Dataset   | Mode                   | AUC-ROC | AUC-PR | ECE    | Mean Spread (Max)
        ----------|------------------------|---------|--------|--------|------------------
        MIMIC-III | deterministic (M=5)    | 0.5528  | 0.1378 | 0.0078 | 0.0780 (0.4504)
        MIMIC-III | bayesian_embeddings    | 0.4924  | 0.1188 | 0.0387 | 0.0108 (0.0617)
        MIMIC-III | bayesian_output        | 0.5400  | 0.1361 | 0.1193 | 0.0315 (0.2316)
        MIMIC-III | bayesian_hidden_output | 0.5259  | 0.1296 | 0.0932 | 0.0566 (0.2997)
        MIMIC-III | fully_bayesian         | 0.5527  | 0.1434 | 0.0299 | 0.1680 (0.4654)
        eICU      | deterministic (M=5)    | 0.6370  | 0.1325 | 0.0042 | 0.0840 (0.7502)
        eICU      | bayesian_embeddings    | 0.5280  | 0.0944 | 0.0082 | 0.0430 (0.2471)
        eICU      | bayesian_output        | 0.6025  | 0.1162 | 0.0786 | 0.0274 (0.4980)
        eICU      | bayesian_hidden_output | 0.6014  | 0.1167 | 0.0643 | 0.0655 (0.7232)
        eICU      | fully_bayesian         | 0.5477  | 0.0965 | 0.0014 | 0.0867 (0.2342)

Key Findings:
    - Scaling ``annealing_steps`` to 50% of total steps and tuning learning rates for
      ``batch_size=32`` lifts deterministic ensemble AUC-ROC on eICU from 0.6058 to
      0.6370 (ECE 0.0042) and on MIMIC-III from 0.5452 to 0.5528 (ECE 0.0078).
    - Even when aggregate calibration is tight (ECE < 0.01), individual patients show
      probability spreads up to 0.7502 on eICU and 0.4504 on MIMIC-III, confirming
      Dusenberry et al.'s finding that population metrics mask patient-level brittleness.
    - On smaller cohorts (MIMIC-III, N=9,583), full KL regularization over large embedding
      tables increases ELBO regularization pressure compared to eICU (N=35,078), where
      fully_bayesian reaches ECE = 0.0014.

Usage:
    # Runnable out-of-the-box with synthetic ICU cohort:
    python examples/mimic3_mortality_uncertainty_rnn.py --dataset synthetic

    # Run on local PhysioNet MIMIC-III or eICU directories:
    python examples/mimic3_mortality_uncertainty_rnn.py --dataset mimic3 --mimic3_root ./mimic3_data
"""

import argparse
import random
from typing import List, Tuple

import numpy as np
import torch

from pyhealth.datasets import (
    MIMIC3Dataset,
    SampleDataset,
    eICUDataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.metrics import binary_metrics_fn
from pyhealth.models import UncertaintyRNN
from pyhealth.models.uncertainty_rnn import VALID_UNCERTAINTY_TYPES
from pyhealth.tasks.mortality_prediction import (
    MortalityPredictionEICU,
    MortalityPredictionMIMIC3,
)

M_CHOICES = [5, 10, 25, 50, 100, 200]


def set_seed(seed: int) -> None:
    """Sets Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_synthetic_cohort(num_patients: int = 120, seed: int = 42) -> SampleDataset:
    """Generates a deterministic synthetic ICU cohort for credential-free runs."""
    rng = random.Random(seed)
    diag_pool = ["401.9", "250.00", "428.0", "410.71", "584.9", "518.81", "038.9"]
    proc_pool = ["96.04", "38.93", "96.71", "87.44", "99.60"]

    samples = []
    for idx in range(num_patients):
        n_visits = rng.randint(1, 4)
        conditions = [
            rng.sample(diag_pool, k=rng.randint(1, 3)) for _ in range(n_visits)
        ]
        procedures = [
            rng.sample(proc_pool, k=rng.randint(1, 2)) for _ in range(n_visits)
        ]
        high_risk = any("038.9" in visit or "410.71" in visit for visit in conditions)
        mortality = 1 if (high_risk and rng.random() < 0.45) or rng.random() < 0.1 else 0
        samples.append(
            {
                "patient_id": f"synth_p{idx}",
                "visit_id": f"synth_v{idx}",
                "conditions": conditions,
                "procedures": procedures,
                "mortality": mortality,
            }
        )

    return SampleDataset(
        samples=samples,
        input_schema={"conditions": "nested_sequence", "procedures": "nested_sequence"},
        output_schema={"mortality": "binary"},
        dataset_name="synthetic_icu",
    )


def build_mortality_dataset(
    dataset_mode: str, mimic3_root: str, eicu_root: str
) -> SampleDataset:
    """Loads synthetic, MIMIC-III, or eICU mortality dataset."""
    if dataset_mode == "synthetic":
        return build_synthetic_cohort()
    if dataset_mode == "mimic3":
        base_dataset = MIMIC3Dataset(
            root=mimic3_root,
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        )
        return base_dataset.set_task(MortalityPredictionMIMIC3())
    if dataset_mode == "eICU":
        base_dataset = eICUDataset(
            root=eicu_root,
            tables=["diagnosis", "medication", "physicalExam"],
        )
        return base_dataset.set_task(MortalityPredictionEICU())
    raise ValueError(f"Unsupported dataset_mode: {dataset_mode}")


def train_single_model(
    dataset: SampleDataset,
    train_loader: torch.utils.data.DataLoader,
    uncertainty_type: str,
    seed: int,
    epochs: int,
    lr: float,
    rnn_dim: int = 256,
    hidden_dim: int = 128,
    device: str = "cuda",
) -> UncertaintyRNN:
    """Trains a single deterministic or Bayesian UncertaintyRNN model."""
    set_seed(seed)
    total_steps = len(train_loader) * epochs
    scaled_annealing_steps = max(1, int(total_steps * 0.5))

    model = UncertaintyRNN(
        dataset=dataset,
        uncertainty_type=uncertainty_type,
        embedding_dim=32,
        rnn_dim=rnn_dim,
        hidden_dim=hidden_dim,
        prior_std=0.292,
        annealing_steps=scaled_annealing_steps,
        num_train_samples=len(train_loader.dataset),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch_idx in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(**batch)
            out["loss"].backward()
            optimizer.step()
            total_loss += out["loss"].item()

        print(
            f"[Seed {seed} | {uncertainty_type}] "
            f"Epoch {epoch_idx}/{epochs} - Train Loss: {total_loss / len(train_loader):.4f}"
        )

    return model


@torch.no_grad()
def predict_bayesian_mc(
    model: UncertaintyRNN,
    data_loader: torch.utils.data.DataLoader,
    num_samples: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draws M Monte Carlo forward passes from a single Bayesian model."""
    model.eval()
    all_mc_probs = []
    y_trues = []

    for m in range(num_samples):
        sample_probs = []
        for batch in data_loader:
            out = model(**batch)
            sample_probs.append(out["y_prob"].cpu().numpy())
            if m == 0:
                y_trues.append(out["y_true"].cpu().numpy())
        all_mc_probs.append(np.concatenate(sample_probs, axis=0))

    return np.stack(all_mc_probs, axis=0), np.concatenate(y_trues, axis=0)


@torch.no_grad()
def predict_ensemble(
    models: List[UncertaintyRNN],
    data_loader: torch.utils.data.DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collects predictions across M deterministic ensemble members."""
    all_member_probs = []
    y_trues = []

    for idx, model in enumerate(models):
        model.eval()
        member_probs = []
        for batch in data_loader:
            out = model(**batch)
            member_probs.append(out["y_prob"].cpu().numpy())
            if idx == 0:
                y_trues.append(out["y_true"].cpu().numpy())
        all_member_probs.append(np.concatenate(member_probs, axis=0))

    return np.stack(all_member_probs, axis=0), np.concatenate(y_trues, axis=0)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(y_true[in_bin])
            conf_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)


def evaluate_uncertainty_distribution(
    prob_matrix: np.ndarray,
    y_true: np.ndarray,
    dataset_name: str,
    label: str = "Model",
) -> None:
    """Prints marginalized discrimination, calibration, and per-patient uncertainty."""
    mean_probs = prob_matrix.mean(axis=0)
    patient_stds = prob_matrix.std(axis=0)
    patient_ranges = prob_matrix.max(axis=0) - prob_matrix.min(axis=0)

    metrics = binary_metrics_fn(
        y_true,
        mean_probs,
        metrics=["pr_auc", "roc_auc", "f1"],
    )
    ece = compute_ece(y_true, mean_probs)

    print(f"\n=== {label} Results (Dataset={dataset_name}, M={prob_matrix.shape[0]}) ===")
    print(f"Marginalized AUC-ROC : {metrics['roc_auc']:.4f}")
    print(f"Marginalized AUC-PR  : {metrics['pr_auc']:.4f}")
    print(f"Calibration Error ECE: {ece:.4f}")
    print(
        f"Mean Patient Std Dev : {patient_stds.mean():.4f} "
        f"(Max: {patient_stds.max():.4f})"
    )
    print(
        f"Mean Patient Spread  : {patient_ranges.mean():.4f} "
        f"(Max Spread: {patient_ranges.max():.4f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate UncertaintyRNN on mortality prediction."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "mimic3", "eICU"],
        help="Dataset source ('synthetic' runs without external PhysioNet files).",
    )
    parser.add_argument("--mimic3_root", type=str, default="./mimic3_data")
    parser.add_argument("--eICU_root", type=str, default="./eICU_data")
    parser.add_argument(
        "--uncertainty_type",
        type=str,
        default="fully_bayesian",
        choices=list(VALID_UNCERTAINTY_TYPES),
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        choices=M_CHOICES,
        help="Number of ensemble members M for deterministic mode.",
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=10,
        choices=M_CHOICES,
        help="Number of Monte Carlo forward passes M for Bayesian modes.",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rnn_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_dataset = build_mortality_dataset(
        args.dataset, args.mimic3_root, args.eICU_root
    )
    print(f"Dataset: {args.dataset} | Samples: {len(sample_dataset)}")

    train_ds, _, test_ds = split_by_patient(
        sample_dataset, [0.4, 0.3, 0.3], seed=args.seed
    )
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    if args.uncertainty_type == "deterministic":
        det_lr = args.lr if args.lr is not None else 1e-4
        ensemble_models = [
            train_single_model(
                sample_dataset,
                train_loader,
                uncertainty_type="deterministic",
                seed=100 + m,
                epochs=args.epochs,
                lr=det_lr,
                rnn_dim=args.rnn_dim,
                hidden_dim=args.hidden_dim,
                device=device,
            )
            for m in range(args.ensemble_size)
        ]
        for sub_m in [m for m in M_CHOICES if m <= args.ensemble_size]:
            prob_matrix, y_true = predict_ensemble(
                ensemble_models[:sub_m], test_loader
            )
            evaluate_uncertainty_distribution(
                prob_matrix,
                y_true,
                dataset_name=args.dataset,
                label=f"Deterministic Ensemble (M={sub_m})",
            )
    else:
        bayes_lr = args.lr if args.lr is not None else 5e-4
        model = train_single_model(
            sample_dataset,
            train_loader,
            uncertainty_type=args.uncertainty_type,
            seed=args.seed,
            epochs=args.epochs,
            lr=bayes_lr,
            rnn_dim=args.rnn_dim,
            hidden_dim=args.hidden_dim,
            device=device,
        )
        prob_matrix, y_true = predict_bayesian_mc(
            model, test_loader, num_samples=args.mc_samples
        )
        evaluate_uncertainty_distribution(
            prob_matrix,
            y_true,
            dataset_name=args.dataset,
            label=args.uncertainty_type,
        )


if __name__ == "__main__":
    main()