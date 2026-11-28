import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.tasks.mortality_prediction import MortalityPredictionMIMIC3
from pyhealth.models import BaseModel
from pyhealth.metrics import binary_metrics_fn

## Default Hyperparameters
random_seed = 12345
batch_size = 256
M_choices = [5, 10, 25, 50, 100, 200]
M_det = M_choices[2]
M_bayes = M_choices[-1]
uncertainty_choices = ["deterministic", "bayesian_embeddings", "bayesian_output", "bayesian_hidden_output", "fully_bayesian"]
uncertainty = uncertainty_choices[2]
epoch = 5


class VariationalEmbedding(nn.Module):
    """Mean-field variational embedding layer with Gaussian prior N(0, prior_std^2)."""
    def __init__(self, num_embeddings, embedding_dim, prior_std=0.292, padding_idx=0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.prior_std = prior_std
        self.padding_idx = padding_idx

        # Learnable posterior parameters: mu and rho (where sigma = softplus(rho))
        self.weight_mu = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight_rho = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight_mu, mean=0.0, std=0.05)
        nn.init.constant_(self.weight_rho, -3.0)

    def forward(self, x):
        if self.training or getattr(self, "mc_sample", True):
            sigma = F.softplus(self.weight_rho)
            eps = torch.randn_like(sigma)
            weight = self.weight_mu + sigma * eps
        else:
            weight = self.weight_mu
        return F.embedding(x, weight, padding_idx=self.padding_idx)

    def kl_divergence(self):
        sigma = F.softplus(self.weight_rho)
        prior_var = self.prior_std ** 2
        kl = (
            torch.log(self.prior_std / sigma)
            + (sigma.pow(2) + self.weight_mu.pow(2)) / (2.0 * prior_var)
            - 0.5
        )
        return kl.sum()


class VariationalLinear(nn.Module):
    """Mean-field variational dense layer with Gaussian prior N(0, prior_std^2)."""
    def __init__(self, in_features, out_features, prior_std=0.292, bias_uncertainty=False):
        super().__init__()
        self.prior_std = prior_std
        self.bias_uncertainty = bias_uncertainty

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -3.0)

        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        if bias_uncertainty:
            self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))
        else:
            self.register_parameter("bias_rho", None)

    def forward(self, x):
        weight_sigma = F.softplus(self.weight_rho)
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        if self.bias_uncertainty:
            bias_sigma = F.softplus(self.bias_rho)
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        sigma = F.softplus(self.weight_rho)
        prior_var = self.prior_std ** 2
        kl = (
            torch.log(self.prior_std / sigma)
            + (sigma.pow(2) + self.weight_mu.pow(2)) / (2.0 * prior_var)
            - 0.5
        ).sum()
        if self.bias_uncertainty:
            b_sigma = F.softplus(self.bias_rho)
            kl += (
                torch.log(self.prior_std / b_sigma)
                + (b_sigma.pow(2) + self.bias_mu.pow(2)) / (2.0 * prior_var)
                - 0.5
            ).sum()
        return kl


class UncertaintyRNN(BaseModel):
    """
    PyHealth 2.0 BaseModel implementation of Dusenberry et al. (2020) RNN supporting:
      - 'deterministic'
      - 'bayesian_embeddings'
      - 'bayesian_output'
      - 'bayesian_hidden_output'
      - 'fully_bayesian'
    """
    def __init__(
        self,
        dataset,
        uncertainty_type="deterministic",
        embedding_dim=32,
        rnn_dim=1024,
        hidden_dim=512,
        prior_std=0.292,
        annealing_steps=972200,
    ):
        # PyHealth 2.0 BaseModel only takes `dataset`
        super().__init__(dataset=dataset)
        self.uncertainty_type = uncertainty_type
        self.annealing_steps = max(1, annealing_steps)
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        # Automatically populated by BaseModel from dataset.input_schema / output_schema
        self.label_key = self.label_keys[0]

        # 1. Embedding Layers using PyHealth 2.0 input_processors
        self.embeddings = nn.ModuleDict()
        use_bayes_emb = uncertainty_type in ["bayesian_embeddings", "fully_bayesian"]

        for feature_key in self.feature_keys:
            processor = self.dataset.input_processors[feature_key]
            vocab_size = len(processor.code_vocab)

            if use_bayes_emb:
                self.embeddings[feature_key] = VariationalEmbedding(
                    vocab_size, embedding_dim, prior_std=prior_std, padding_idx=0
                )
            else:
                self.embeddings[feature_key] = nn.Embedding(
                    vocab_size, embedding_dim, padding_idx=0
                )

        total_emb_dim = len(self.feature_keys) * embedding_dim

        # 2. Sequence Encoder (LSTM)
        self.lstm = nn.LSTM(
            input_size=total_emb_dim,
            hidden_size=rnn_dim,
            num_layers=1,
            batch_first=True,
        )

        # 3. Hidden & Output Dense Layers (Table 6 architecture)
        use_bayes_hidden = uncertainty_type in ["bayesian_hidden_output", "fully_bayesian"]
        use_bayes_out = uncertainty_type in [
            "bayesian_output",
            "bayesian_hidden_output",
            "fully_bayesian",
        ]

        self.hidden_layer = (
            VariationalLinear(rnn_dim, hidden_dim, prior_std=prior_std)
            if use_bayes_hidden
            else nn.Linear(rnn_dim, hidden_dim)
        )
        self.output_layer = (
            VariationalLinear(hidden_dim, 1, prior_std=prior_std)
            if use_bayes_out
            else nn.Linear(hidden_dim, 1)
        )

    def total_kl_divergence(self):
        kl = torch.tensor(0.0, device=self.global_step.device)
        for module in self.modules():
            if hasattr(module, "kl_divergence"):
                kl = kl + module.kl_divergence()
        return kl

    def encode_sequence(self, **kwargs):
        """Embeds and concatenates feature tensors with mismatched sequence lengths."""
        raw_tokens = {}
        max_len = 0

        # 1. Collect tokens and find the maximum sequence length across all features
        for feature_key in self.feature_keys:
            tokens = kwargs[feature_key].to(self.device).long()
            if tokens.dim() == 2:
                # Shape: [batch, seq_len] -> [batch, seq_len, 1]
                tokens = tokens.unsqueeze(-1)
            raw_tokens[feature_key] = tokens
            max_len = max(max_len, tokens.size(1))

        embedded_features = []
        combined_mask = None

        # 2. Right-pad shorter feature tensors along dim=1 to match max_len
        for feature_key in self.feature_keys:
            tokens = raw_tokens[feature_key]
            pad_len = max_len - tokens.size(1)
            if pad_len > 0:
                # Pad dim=1 (sequence length) on the right with padding_idx=0
                tokens = F.pad(tokens, (0, 0, 0, pad_len), value=0)

            step_mask = (tokens.sum(dim=-1) != 0).long()
            combined_mask = step_mask if combined_mask is None else torch.maximum(combined_mask, step_mask)

            # Embed and sum across inner code dimension -> [batch, max_len, embedding_dim]
            emb = self.embeddings[feature_key](tokens).sum(dim=-2)
            embedded_features.append(emb)

        x = torch.cat(embedded_features, dim=-1)  # [batch, max_len, total_emb_dim]
        rnn_out, _ = self.lstm(x)

        # Extract final valid timestep per patient
        lengths = combined_mask.sum(dim=-1).clamp(min=1) - 1
        batch_indices = torch.arange(rnn_out.size(0), device=self.device)
        final_hidden = rnn_out[batch_indices, lengths]
        return final_hidden

    def forward(self, **kwargs):
        h = self.encode_sequence(**kwargs)
        h = F.relu(self.hidden_layer(h))
        logits = self.output_layer(h).squeeze(-1)

        y_true = kwargs[self.label_key].to(self.device).float().view(-1)
        nll_loss = F.binary_cross_entropy_with_logits(logits, y_true)

        # Variational Inference Objective: NLL + annealed KL / N_train
        if self.uncertainty_type != "deterministic":
            if self.training:
                self.global_step += 1
            kl_weight = min(1.0, float(self.global_step) / self.annealing_steps)
            num_train_samples = getattr(self, "num_train_samples", 30000)
            kl_loss = (self.total_kl_divergence() / num_train_samples) * kl_weight
            loss = nll_loss + kl_loss
        else:
            loss = nll_loss

        y_prob = torch.sigmoid(logits)
        return {"loss": loss, "y_prob": y_prob, "y_true": y_true}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_single_model(
    dataset,
    train_loader,
    val_loader,
    uncertainty_type,
    seed,
    epochs=10,
    lr=1.238e-3,
    device="cuda",
):
    set_seed(seed)
    model = UncertaintyRNN(
        dataset=dataset,
        uncertainty_type=uncertainty_type,
        embedding_dim=32,
        rnn_dim=1024,
        hidden_dim=512,
        prior_std=0.292,
        annealing_steps=972200,
    ).to(device)
    model.num_train_samples = len(train_loader.dataset)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
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
            f"Epoch {epoch}/{epochs} - Train Loss: {total_loss / len(train_loader):.4f}"
        )

    return model


@torch.no_grad()
def predict_bayesian_mc(model, data_loader, num_samples=200):
    """Draws M=200 Monte Carlo forward passes from a single Bayesian model."""
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

    # Shape: [M, N_patients]
    mc_matrix = np.stack(all_mc_probs, axis=0)
    y_true = np.concatenate(y_trues, axis=0)
    return mc_matrix, y_true


@torch.no_grad()
def predict_ensemble(models, data_loader):
    """Collects predictions across M deterministic ensemble models."""
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

    # Shape: [M, N_patients]
    ensemble_matrix = np.stack(all_member_probs, axis=0)
    y_true = np.concatenate(y_trues, axis=0)
    return ensemble_matrix, y_true


def compute_ece(y_true, y_prob, n_bins=10):
    """Computes Expected Calibration Error (Equation in Section 2.3)."""
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


def evaluate_uncertainty_distribution(prob_matrix, y_true, label="Model"):
    """Computes population metrics (Table 1/2) AND patient-level spread (Figure 1/3)."""
    mean_probs = prob_matrix.mean(axis=0)
    patient_stds = prob_matrix.std(axis=0)
    patient_ranges = prob_matrix.max(axis=0) - prob_matrix.min(axis=0)

    metrics = binary_metrics_fn(
        y_true,
        mean_probs,
        metrics=["pr_auc", "roc_auc", "f1"],
    )
    ece = compute_ece(y_true, mean_probs)

    print(f"\n=== {label} Results (M={prob_matrix.shape[0]}) ===")
    print(f"Marginalized AUC-ROC : {metrics['roc_auc']:.4f}")
    print(f"Marginalized AUC-PR  : {metrics['pr_auc']:.4f}")
    print(f"Calibration Error ECE: {ece:.4f}")
    print(f"Mean Patient Std Dev : {patient_stds.mean():.4f} (Max: {patient_stds.max():.4f})")
    print(f"Mean Patient Spread  : {patient_ranges.mean():.4f} (Max Spread: {patient_ranges.max():.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mimic3_root",
        type=str,
        default="./data",
        help="Path to MIMIC-III CSVs"
    )
    parser.add_argument(
        "--uncertainty_type",
        type=str,
        default=uncertainty,
        choices=["deterministic", "bayesian_embeddings", "bayesian_output", "bayesian_hidden_output", "fully_bayesian"],
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=M_det,
        choices=M_choices,
        help="M models for deterministic ensemble"
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=M_bayes,
        choices=M_choices,
        help="M samples for Bayesian RNN"
    )
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--epochs", type=int, default=epoch)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load MIMIC-III via PyHealth (replaces missing `constants` and `util`)
    base_dataset = MIMIC3Dataset(
        root=args.mimic3_root,
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"]
    )
    sample_dataset = base_dataset.set_task(MortalityPredictionMIMIC3())

    print(len(sample_dataset))
    print(sample_dataset[0])
    print(sample_dataset.input_schema)
    print(sample_dataset.output_schema)

    # 4:3:3 patient-level split matching
    train_ds, val_ds, test_ds = split_by_patient(sample_dataset, [0.4, 0.3, 0.3], seed=random_seed)

    def count_pos(ds):
        ys = [batch["mortality"] if "mortality" in batch else batch["label"] for batch in
              get_dataloader(ds, batch_size=batch_size, shuffle=False)]
        y_all = torch.cat([t.view(-1) for t in ys]).numpy()
        return int(y_all.sum()), len(y_all)

    sample_p, sample_all = count_pos(sample_dataset)
    train_p, train_all = count_pos(train_ds)
    val_p, val_all = count_pos(val_ds)
    test_p, test_all = count_pos(test_ds)

    print(f"Sample Data Positive Count: {sample_p} out of {sample_all}")
    print(f"Train Data Positive Count: {train_p} out of {train_all}")
    print(f"Validation Data Positive Count: {val_p} out of {val_all}")
    print(f"Test Data Positive Count: {test_p} out of {test_all}")

    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Run either Deterministic Deep Ensemble OR Single Bayesian RNN
    if args.uncertainty_type == "deterministic":
        ensemble_models = []
        for m in range(args.ensemble_size):
            seed = 100 + m
            model = train_single_model(
                sample_dataset,
                train_loader,
                val_loader,
                uncertainty_type="deterministic",
                seed=seed,
                epochs=args.epochs,
                lr=3.035e-4,  # Table 6 deterministic LR
                device=device,
            )
            ensemble_models.append(model)

        i = 0
        while M_choices[i] <= M_det:
            prob_matrix, y_true = predict_ensemble(ensemble_models[:M_choices[i]], test_loader)
            evaluate_uncertainty_distribution(prob_matrix, y_true, label=f"Deterministic Ensemble with M={M_choices[i]}")
            i += 1
    else:
        model = train_single_model(
            sample_dataset,
            train_loader,
            val_loader,
            uncertainty_type=args.uncertainty_type,
            seed=random_seed,
            epochs=args.epochs,
            lr=1.238e-3,  # Table 6 Bayesian Embeddings LR
            device=device,
        )
        prob_matrix, y_true = predict_bayesian_mc(model, test_loader, num_samples=args.mc_samples)
        evaluate_uncertainty_distribution(prob_matrix, y_true, label=args.uncertainty_type)


if __name__ == "__main__":
    main()