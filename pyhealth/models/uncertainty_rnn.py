"""Uncertainty-aware RNN models for Electronic Health Records (EHR).

Reference:
    M. W. Dusenberry, D. Tran, E. Choi, J. Kemp, J. Nixon, G. Jerfel,
    K. Heller, and A. M. Dai, "Analyzing the Role of Model Uncertainty
    for Electronic Health Records," in Proc. ACM Conference on Health,
    Inference, and Learning (CHIL '20), 2020, pp. 204-213.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

VALID_UNCERTAINTY_TYPES = (
    "deterministic",
    "bayesian_embeddings",
    "bayesian_output",
    "bayesian_hidden_output",
    "fully_bayesian",
)


class VariationalEmbedding(nn.Module):
    """Mean-field variational embedding layer with a Gaussian prior.

    Places a factorized normal posterior ``N(weight_mu, softplus(weight_rho)^2)``
    over discrete code embeddings with zero-mean Gaussian prior ``N(0, prior_std^2)``.

    Args:
        num_embeddings: Size of the vocabulary dictionary.
        embedding_dim: Dimensionality of each embedding vector.
        prior_std: Standard deviation of the isotropic Gaussian prior.
        padding_idx: Index whose embedding gradient is masked out during padding.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        prior_std: float = 0.292,
        padding_idx: int = 0,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Samples embeddings via reparameterization and looks up indices.

        Args:
            x: Integer token tensor of shape ``[batch_size, ...]``.

        Returns:
            Embedded tensor of shape ``[batch_size, ..., embedding_dim]``.
        """
        if self.training or getattr(self, "mc_sample", True):
            sigma = F.softplus(self.weight_rho)
            weight = self.weight_mu + sigma * torch.randn_like(sigma)
        else:
            weight = self.weight_mu
        return F.embedding(x, weight, padding_idx=self.padding_idx)

    def kl_divergence(self) -> torch.Tensor:
        """Computes analytical KL divergence ``KL(q(W) || p(W))``."""
        sigma = F.softplus(self.weight_rho)
        prior_var = self.prior_std ** 2
        kl = (
            torch.log(self.prior_std / sigma)
            + (sigma.pow(2) + self.weight_mu.pow(2)) / (2.0 * prior_var)
            - 0.5
        )
        return kl.sum()


class VariationalLinear(nn.Module):
    """Mean-field variational linear layer with a Gaussian prior.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        prior_std: Standard deviation of the Gaussian weight prior.
        bias_uncertainty: Whether to place a variational posterior on bias terms.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies an affine transformation using sampled variational weights.

        Args:
            x: Input tensor of shape ``[batch_size, in_features]``.

        Returns:
            Output tensor of shape ``[batch_size, out_features]``.
        """
        weight_sigma = F.softplus(self.weight_rho)
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        if self.bias_uncertainty:
            bias_sigma = F.softplus(self.bias_rho)
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """Computes analytical KL divergence ``KL(q(W, b) || p(W, b))``."""
        sigma = F.softplus(self.weight_rho)
        prior_var = self.prior_std ** 2
        kl = (
            torch.log(self.prior_std / sigma)
            + (sigma.pow(2) + self.weight_mu.pow(2)) / (2.0 * prior_var)
            - 0.5
        ).sum()
        if self.bias_uncertainty and self.bias_rho is not None:
            b_sigma = F.softplus(self.bias_rho)
            kl += (
                torch.log(self.prior_std / b_sigma)
                + (b_sigma.pow(2) + self.bias_mu.pow(2)) / (2.0 * prior_var)
                - 0.5
            ).sum()
        return kl


class UncertaintyRNN(BaseModel):
    """Recurrent neural network with deterministic and Bayesian uncertainty modes.

    Implements the clinical sequence architecture of Dusenberry et al. (2020),
    supporting deterministic deep ensembles as well as four variational Bayesian
    parameterizations trained via ELBO minimization with linear KL annealing.

    Args:
        dataset: Processed PyHealth ``SampleDataset`` defining input/output schemas.
        uncertainty_type: Uncertainty regime to instantiate. Must be one of
            ``"deterministic"``, ``"bayesian_embeddings"``, ``"bayesian_output"``,
            ``"bayesian_hidden_output"``, or ``"fully_bayesian"``.
        embedding_dim: Dimension of per-feature code embeddings. Default is 32.
        rnn_dim: Hidden state size of the LSTM sequence encoder. Default is 1024.
        hidden_dim: Width of the post-LSTM dense hidden layer. Default is 512.
        prior_std: Prior standard deviation for variational layers. Default is 0.292.
        annealing_steps: Number of training steps over which KL weight ramps from
            0.0 to 1.0. Default is 972200.
        num_train_samples: Training set size used to scale the KL regularization
            term in the ELBO objective. If ``None``, defaults to ``len(dataset)``.

    Examples:
        >>> from pyhealth.models import UncertaintyRNN
        >>> model = UncertaintyRNN(
        ...     dataset=sample_dataset,
        ...     uncertainty_type="bayesian_embeddings",
        ...     embedding_dim=32,
        ...     rnn_dim=256,
        ...     hidden_dim=128,
        ... )
        >>> outputs = model(**batch)
        >>> loss, y_prob = outputs["loss"], outputs["y_prob"]
    """

    def __init__(
        self,
        dataset: SampleDataset,
        uncertainty_type: str = "deterministic",
        embedding_dim: int = 32,
        rnn_dim: int = 1024,
        hidden_dim: int = 512,
        prior_std: float = 0.292,
        annealing_steps: int = 972200,
        num_train_samples: Optional[int] = None,
    ) -> None:
        super().__init__(dataset=dataset)
        if uncertainty_type not in VALID_UNCERTAINTY_TYPES:
            raise ValueError(
                f"Unsupported uncertainty_type '{uncertainty_type}'. "
                f"Choose from {VALID_UNCERTAINTY_TYPES}."
            )

        self.uncertainty_type = uncertainty_type
        self.embedding_dim = embedding_dim
        self.rnn_dim = rnn_dim
        self.hidden_dim = hidden_dim
        self.prior_std = prior_std
        self.annealing_steps = max(1, annealing_steps)
        self.num_train_samples = (
            num_train_samples if num_train_samples is not None else max(1, len(dataset))
        )
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        self.label_key = self.label_keys[0]

        # 1. Feature embedding layers
        self.embeddings = nn.ModuleDict()
        use_bayes_emb = uncertainty_type in ("bayesian_embeddings", "fully_bayesian")

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

        # 2. Sequence encoder (LSTM)
        self.lstm = nn.LSTM(
            input_size=total_emb_dim,
            hidden_size=rnn_dim,
            num_layers=1,
            batch_first=True,
        )

        # 3. Hidden and output dense layers
        use_bayes_hidden = uncertainty_type in (
            "bayesian_hidden_output",
            "fully_bayesian",
        )
        use_bayes_out = uncertainty_type in (
            "bayesian_output",
            "bayesian_hidden_output",
            "fully_bayesian",
        )

        self.hidden_layer: nn.Module = (
            VariationalLinear(rnn_dim, hidden_dim, prior_std=prior_std)
            if use_bayes_hidden
            else nn.Linear(rnn_dim, hidden_dim)
        )
        self.output_layer: nn.Module = (
            VariationalLinear(hidden_dim, 1, prior_std=prior_std)
            if use_bayes_out
            else nn.Linear(hidden_dim, 1)
        )

    def total_kl_divergence(self) -> torch.Tensor:
        """Aggregates KL divergence across all variational submodules."""
        kl = torch.tensor(0.0, device=self.global_step.device)
        for module in self.modules():
            if hasattr(module, "kl_divergence"):
                kl = kl + module.kl_divergence()
        return kl

    def encode_sequence(self, **kwargs: Any) -> torch.Tensor:
        """Embeds, aligns, and encodes variable-length clinical feature sequences.

        Args:
            **kwargs: Batch dictionary mapping each ``feature_key`` to an integer
                tensor of shape ``[batch_size, seq_len]`` or
                ``[batch_size, seq_len, max_codes]``.

        Returns:
            Final valid LSTM hidden state tensor of shape ``[batch_size, rnn_dim]``.
        """
        raw_tokens: Dict[str, torch.Tensor] = {}
        max_len = 0

        for feature_key in self.feature_keys:
            tokens = kwargs[feature_key].to(self.device).long()
            if tokens.dim() == 2:
                tokens = tokens.unsqueeze(-1)
            raw_tokens[feature_key] = tokens
            max_len = max(max_len, tokens.size(1))

        embedded_features = []
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

            emb = self.embeddings[feature_key](tokens).sum(dim=-2)
            embedded_features.append(emb)

        x = torch.cat(embedded_features, dim=-1)
        rnn_out, _ = self.lstm(x)

        assert combined_mask is not None
        lengths = combined_mask.sum(dim=-1).clamp(min=1) - 1
        batch_indices = torch.arange(rnn_out.size(0), device=self.device)
        return rnn_out[batch_indices, lengths]

    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Runs a forward pass and computes NLL or annealed ELBO loss.

        Args:
            **kwargs: Batch dictionary containing all ``feature_keys`` and
                ``label_key``.

        Returns:
            Dictionary with keys ``"loss"``, ``"y_prob"``, and ``"y_true"``.
        """
        h = self.encode_sequence(**kwargs)
        h = F.relu(self.hidden_layer(h))
        logits = self.output_layer(h).squeeze(-1)

        y_true = kwargs[self.label_key].to(self.device).float().view(-1)
        nll_loss = F.binary_cross_entropy_with_logits(logits, y_true)

        if self.uncertainty_type != "deterministic":
            if self.training:
                self.global_step += 1
            kl_weight = min(1.0, float(self.global_step) / self.annealing_steps)
            kl_loss = (self.total_kl_divergence() / self.num_train_samples) * kl_weight
            loss = nll_loss + kl_loss
        else:
            loss = nll_loss

        y_prob = torch.sigmoid(logits)
        return {"loss": loss, "y_prob": y_prob, "y_true": y_true}