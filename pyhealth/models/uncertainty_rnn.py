# Author: Donggyu Lee, Jinfeng Shen
# Paper: Analyzing the Role of Model Uncertainty for Electronic Health Records
# Link: https://arxiv.org/abs/1906.03842
# Description: Variational Bayesian RNN (UncertaintyRNN) supporting deterministic,
#              Bayesian embeddings, Bayesian output, and Bayesian hidden+output
#              configurations for quantifying epistemic uncertainty on EHR data.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class VariationalEmbedding(nn.Module):
    """Mean-field variational embedding layer with Gaussian prior N(0, prior_std^2).

    Args:
        num_embeddings: Size of the token vocabulary.
        embedding_dim: Dimensionality of each embedding vector.
        prior_std: Standard deviation of the zero-mean isotropic Gaussian prior.
        padding_idx: Token index excluded from sampling and KL regularization.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        prior_std: float = 0.292,
        padding_idx: Optional[int] = 0,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.prior_std = prior_std
        self.padding_idx = padding_idx

        self.weight_mu = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight_rho = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight_mu, mean=0.0, std=0.05)
        nn.init.constant_(self.weight_rho, -3.0)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight_mu[self.padding_idx].fill_(0.0)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            sigma = F.softplus(self.weight_rho)
            eps = torch.randn_like(sigma)
            weight = self.weight_mu + sigma * eps
        else:
            weight = self.weight_mu

        if self.padding_idx is not None:
            weight = weight.clone()
            weight[self.padding_idx] = 0.0

        return F.embedding(x, weight, padding_idx=self.padding_idx)

    def kl_divergence(self) -> torch.Tensor:
        """Compute exact analytical KL[q(W) || p(W)] excluding padding_idx."""
        sigma = F.softplus(self.weight_rho)
        prior_var = self.prior_std ** 2
        kl = (
            torch.log(self.prior_std / sigma)
            + (sigma.pow(2) + self.weight_mu.pow(2)) / (2.0 * prior_var)
            - 0.5
        )
        if self.padding_idx is not None:
            mask = torch.ones(
                self.num_embeddings, device=self.weight_mu.device, dtype=torch.bool
            )
            mask[self.padding_idx] = False
            kl = kl[mask]
        return kl.sum()


class VariationalLinear(nn.Module):
    """Mean-field variational dense layer with Gaussian prior N(0, prior_std^2).

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        prior_std: Standard deviation of the zero-mean Gaussian prior.
        bias_uncertainty: Whether to place a variational posterior on the bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 0.292,
        bias_uncertainty: bool = False,
    ) -> None:
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

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            weight_sigma = F.softplus(self.weight_rho)
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            if self.bias_uncertainty:
                bias_sigma = F.softplus(self.bias_rho)
                bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
            else:
                bias = self.bias_mu
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
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
    """PyHealth BaseModel implementation of Dusenberry et al. (ACM CHIL 2020) RNN.

    Supports deterministic training as well as variational Bayesian inference over
    embeddings, output layer, or hidden + output layers.

    Args:
        dataset: PyHealth SampleDataset defining feature and label schemas.
        uncertainty_type: One of ``"deterministic"``, ``"bayesian_embeddings"``,
            ``"bayesian_output"``, ``"bayesian_hidden_output"``, or
            ``"fully_bayesian"``.
        embedding_dim: Dimensionality of token embeddings.
        rnn_dim: Hidden dimension of the LSTM encoder.
        hidden_dim: Hidden dimension of the post-LSTM dense layer.
        prior_std: Prior Gaussian standard deviation for variational weights.
        annealing_steps: Number of training steps over which KL weight anneals
            linearly from 0 to 1.
        num_train_samples: Training set size N used to scale KL / N.
    """

    VALID_TYPES = {
        "deterministic",
        "bayesian_embeddings",
        "bayesian_output",
        "bayesian_hidden_output",
        "fully_bayesian",
    }

    def __init__(
        self,
        dataset: SampleDataset,
        uncertainty_type: str = "bayesian_embeddings",
        embedding_dim: int = 32,
        rnn_dim: int = 1024,
        hidden_dim: int = 512,
        prior_std: float = 0.292,
        annealing_steps: int = 972200,
        num_train_samples: int = 30000,
    ) -> None:
        super().__init__(dataset=dataset)
        if uncertainty_type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid uncertainty_type '{uncertainty_type}'. "
                f"Expected one of {sorted(self.VALID_TYPES)}."
            )

        self.uncertainty_type = uncertainty_type
        self.annealing_steps = max(1, annealing_steps)
        self.num_train_samples = max(1, num_train_samples)
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        assert len(self.label_keys) == 1, "Only single label key is supported."
        self.label_key = self.label_keys[0]
        self.output_size = self.get_output_size()

        self.embeddings = nn.ModuleDict()
        use_bayes_emb = uncertainty_type in {"bayesian_embeddings", "fully_bayesian"}

        for feature_key in self.feature_keys:
            processor = self.dataset.input_processors[feature_key]
            vocab_map = getattr(processor, "code_vocab", None) or getattr(
                processor, "vocab", None
            )
            vocab_size = len(vocab_map)

            if use_bayes_emb:
                self.embeddings[feature_key] = VariationalEmbedding(
                    vocab_size, embedding_dim, prior_std=prior_std, padding_idx=0
                )
            else:
                self.embeddings[feature_key] = nn.Embedding(
                    vocab_size, embedding_dim, padding_idx=0
                )

        total_emb_dim = len(self.feature_keys) * embedding_dim

        self.lstm = nn.LSTM(
            input_size=total_emb_dim,
            hidden_size=rnn_dim,
            num_layers=1,
            batch_first=True,
        )

        use_bayes_hidden = uncertainty_type in {
            "bayesian_hidden_output",
            "fully_bayesian",
        }
        use_bayes_out = uncertainty_type in {
            "bayesian_output",
            "bayesian_hidden_output",
            "fully_bayesian",
        }

        self.hidden_layer = (
            VariationalLinear(rnn_dim, hidden_dim, prior_std=prior_std)
            if use_bayes_hidden
            else nn.Linear(rnn_dim, hidden_dim)
        )
        self.output_layer = (
            VariationalLinear(hidden_dim, self.output_size, prior_std=prior_std)
            if use_bayes_out
            else nn.Linear(hidden_dim, self.output_size)
        )

    def total_kl_divergence(self) -> torch.Tensor:
        """Sum analytical KL divergence across all variational modules."""
        kl = torch.tensor(0.0, device=self.global_step.device)
        for module in self.modules():
            if hasattr(module, "kl_divergence"):
                kl = kl + module.kl_divergence()
        return kl

    def encode_sequence(self, sample: bool = True, **kwargs) -> torch.Tensor:
        """Embed and concatenate multi-feature sequences with mismatched lengths."""
        raw_tokens: Dict[str, torch.Tensor] = {}
        max_len = 0

        for feature_key in self.feature_keys:
            tokens = kwargs[feature_key]
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
            else:
                tokens = tokens.to(device=self.device, dtype=torch.long)

            if tokens.dim() == 2:
                tokens = tokens.unsqueeze(-1)
            raw_tokens[feature_key] = tokens
            max_len = max(max_len, tokens.size(1))

        embedded_features: List[torch.Tensor] = []
        combined_mask: Optional[torch.Tensor] = None

        for feature_key in self.feature_keys:
            tokens = raw_tokens[feature_key]
            pad_len = max_len - tokens.size(1)
            if pad_len > 0:
                tokens = F.pad(tokens, (0, 0, 0, pad_len), value=0)

            step_mask = (tokens.sum(dim=-1) != 0).long()
            combined_mask = (
                step_mask
                if combined_mask is None
                else torch.maximum(combined_mask, step_mask)
            )

            emb_layer = self.embeddings[feature_key]
            if isinstance(emb_layer, VariationalEmbedding):
                emb = emb_layer(tokens, sample=sample).sum(dim=-2)
            else:
                emb = emb_layer(tokens).sum(dim=-2)
            embedded_features.append(emb)

        x = torch.cat(embedded_features, dim=-1)
        rnn_out, _ = self.lstm(x)

        lengths = combined_mask.sum(dim=-1).clamp(min=1) - 1
        batch_indices = torch.arange(rnn_out.size(0), device=self.device)
        return rnn_out[batch_indices, lengths]

    def _forward_logits(self, sample: bool = True, **kwargs) -> torch.Tensor:
        h = self.encode_sequence(sample=sample, **kwargs)
        if isinstance(self.hidden_layer, VariationalLinear):
            h = F.relu(self.hidden_layer(h, sample=sample))
        else:
            h = F.relu(self.hidden_layer(h))

        if isinstance(self.output_layer, VariationalLinear):
            return self.output_layer(h, sample=sample)
        return self.output_layer(h)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute task loss, annealed KL divergence, and predicted probabilities."""
        logits = self._forward_logits(sample=True, **kwargs)
        y_true = kwargs[self.label_key].to(self.device)
        loss_fn = self.get_loss_function()
        nll_loss = loss_fn(logits, y_true)

        if self.uncertainty_type != "deterministic":
            if self.training:
                self.global_step += 1
            kl_weight = min(1.0, float(self.global_step) / self.annealing_steps)
            kl_loss = (self.total_kl_divergence() / self.num_train_samples) * kl_weight
            loss = nll_loss + kl_loss
        else:
            kl_loss = torch.tensor(0.0, device=self.device)
            loss = nll_loss

        y_prob = self.prepare_y_prob(logits)
        return {
            "loss": loss,
            "nll_loss": nll_loss,
            "kl_loss": kl_loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }

    @torch.no_grad()
    def predict_uncertainty(
        self, num_samples: int = 50, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Draw M Monte Carlo forward passes to quantify patient-level uncertainty."""
        preds: List[torch.Tensor] = []
        for _ in range(num_samples):
            logits = self._forward_logits(sample=True, **kwargs)
            preds.append(self.prepare_y_prob(logits))

        stacked = torch.stack(preds, dim=0)
        return {
            "samples": stacked,
            "mean": stacked.mean(dim=0),
            "std": stacked.std(dim=0),
            "range": stacked.max(dim=0).values - stacked.min(dim=0).values,
        }