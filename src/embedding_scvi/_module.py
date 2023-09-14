from __future__ import annotations

from dataclasses import field

import chex
import numpy as np
import torch
import torch.distributions as dist
from scvi import REGISTRY_KEYS
from scvi._types import LossRecord, Tensor
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch import nn

from ._components import ExtendableEmbeddingList, MLPMultiOutput
from ._constants import TENSORS_KEYS


@chex.dataclass
class LossOutput(LossOutput):
    loss: LossRecord
    reconstruction_loss: LossRecord | None = None
    kl_local: LossRecord | None = None
    kl_global: LossRecord | None = None
    classification_loss: LossRecord | None = None
    logits: Tensor | None = None
    true_labels: Tensor | None = None
    extra_metrics: dict[str, Tensor] | None = field(default_factory=dict)
    n_obs_minibatch: int | None = None
    reconstruction_loss_sum: Tensor = field(default=None, init=False)
    kl_local_sum: Tensor = field(default=None, init=False)
    kl_global_sum: Tensor = field(default=None, init=False)
    extra_tensors: dict[str, Tensor] | None = field(default_factory=dict)


class EmbeddingVAE(BaseModuleClass):
    def __init__(
        self,
        n_vars: int,
        n_latent: int = 25,
        categorical_covariates: list[int] | None = None,
        likelihood: str = "zinb",
        encoder_kwargs: dict | None = None,
        decoder_kwargs: dict | None = None,
    ):
        super().__init__()

        self.n_vars = n_vars
        self.n_latent = n_latent
        self.categorical_covariates = categorical_covariates
        self.likelihood = likelihood
        self.encoder_kwargs = encoder_kwargs or {}
        self.decoder_kwargs = decoder_kwargs or {}

        if likelihood == "zinb":
            decoder_n_out_params = 3
            # scale, r, dropout
            decoder_param_activations = ["softmax", None, None]
        elif likelihood == "nb":
            decoder_n_out_params = 3
            # mu, theta, scale
            decoder_param_activations = [None, None, "softplus"]
        elif likelihood == "poisson":
            decoder_n_out_params = 2
            # mu, scale
            decoder_param_activations = [None, "softplus"]
        else:
            raise ValueError(f"Invalid likelihood {likelihood}")

        _encoder_kwargs = {
            "n_hidden": 256,
            "n_layers": 2,
            "bias": True,
            "norm": "layer",
            "activation": "relu",
            "dropout_rate": 0.1,
            "residual": True,
        }
        _encoder_kwargs.update(self.encoder_kwargs)
        self.encoder = MLPMultiOutput(
            n_in=self.n_vars,
            n_out=self.n_latent,
            n_out_params=2,
            param_activations=[None, "softplus"],
            **_encoder_kwargs,
        )

        _decoder_kwargs = {
            "n_hidden": 256,
            "n_layers": 2,
            "bias": True,
            "norm": "layer",
            "activation": "relu",
            "dropout_rate": None,
            "residual": True,
        }
        _decoder_kwargs.update(self.decoder_kwargs)
        self.decoder = MLPMultiOutput(
            n_in=self.n_latent,
            n_out=self.n_vars,
            n_out_params=decoder_n_out_params,
            param_activations=decoder_param_activations,
            **_decoder_kwargs,
        )

        self.covariates_encoder = nn.Identity()
        if self.categorical_covariates is not None:
            self.covariates_encoder = ExtendableEmbeddingList(
                num_embeddings=self.categorical_covariates,
                embedding_dim=self.n_latent,
            )

    def get_covariate_embeddings(
        self,
        covariate_indexes: list[int] | int | None,
        as_tensor: bool = True,
    ) -> list[torch.Tensor | np.ndarray]:
        if isinstance(covariate_indexes, int):
            covariate_indexes = [covariate_indexes]
        elif covariate_indexes is None:
            covariate_indexes = list(range(len(self.categorical_covariates)))

        weights = [self.covariates_encoder.get_embedding_weight(i, as_tensor=as_tensor) for i in covariate_indexes]
        return weights

    def _get_inference_input(self, tensors: dict[str, torch.Tensor]) -> dict:
        x = tensors[REGISTRY_KEYS.X_KEY]
        covariates = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY, None)
        return {
            REGISTRY_KEYS.X_KEY: x,
            REGISTRY_KEYS.CAT_COVS_KEY: covariates,
        }

    @auto_move_data
    def inference(
        self,
        X: torch.Tensor,
        extra_categorical_covs: torch.Tensor | None = None,
        subset_categorical_covs: int | list[int] | None = None,
    ):
        X = torch.log1p(X)
        library_size = torch.log(X.sum(dim=1, keepdim=True))

        posterior_loc, posterior_scale = self.encoder(X)
        posterior = dist.Normal(posterior_loc, posterior_scale + 1e-9)
        prior = dist.Normal(torch.zeros_like(posterior_loc), torch.ones_like(posterior_scale))
        z = posterior.rsample()

        covariates_z = self.covariates_encoder(
            extra_categorical_covs,
            subset=subset_categorical_covs,
        )

        return {
            TENSORS_KEYS.Z_KEY: z,
            TENSORS_KEYS.QZ_KEY: posterior,
            TENSORS_KEYS.PZ_KEY: prior,
            TENSORS_KEYS.COVARIATES_Z_KEY: covariates_z,
            REGISTRY_KEYS.OBSERVED_LIB_SIZE: library_size,
        }

    def _get_generative_input(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
    ) -> dict:
        z = inference_outputs[TENSORS_KEYS.Z_KEY]
        covariates_z = inference_outputs[TENSORS_KEYS.COVARIATES_Z_KEY]
        library_size = inference_outputs[REGISTRY_KEYS.OBSERVED_LIB_SIZE]

        return {
            TENSORS_KEYS.Z_KEY: z,
            TENSORS_KEYS.COVARIATES_Z_KEY: covariates_z,
            REGISTRY_KEYS.OBSERVED_LIB_SIZE: library_size,
        }

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        covariates_z: torch.Tensor | None = None,
        observed_lib_size: torch.Tensor | None = None,
    ):
        if covariates_z is not None:
            # (1, n_obs, n_latent), (n_covariates, n_obs, n_latent)
            # -> (n_covariates + 1, n_obs, n_latent)
            z_covariates = torch.cat([z[None, :], covariates_z], dim=0)
            # (n_covariates + 1, n_obs, n_latent) -> (n_obs, n_latent)
            z = torch.sum(z_covariates, dim=0)

        likelihood_params = self.decoder(z)
        if self.likelihood == "zinb":
            scale, r, dropout = likelihood_params
            rate = torch.exp(observed_lib_size) * scale
            r = torch.exp(r)
            px = ZeroInflatedNegativeBinomial(
                mu=rate,
                theta=r,
                zi_logits=dropout,
                scale=scale,
            )
        elif self.likelihood == "nb":
            mu, theta, scale = likelihood_params
            px = NegativeBinomial(
                mu=mu,
                theta=theta,
                scale=scale,
            )
        elif self.likelihood == "poisson":
            mu, scale = likelihood_params
            px = Poisson(
                rate=mu,
                scale=scale,
            )

        return {
            TENSORS_KEYS.PX_KEY: px,
        }

    @auto_move_data
    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
        kl_weight: float = 1.0,
    ) -> LossOutput:
        X = tensors[REGISTRY_KEYS.X_KEY]
        posterior = inference_outputs[TENSORS_KEYS.QZ_KEY]
        prior = inference_outputs[TENSORS_KEYS.PZ_KEY]
        likelihood = generative_outputs[TENSORS_KEYS.PX_KEY]

        # (n_obs, n_latent) -> (n_obs,)
        kl_div = dist.kl_divergence(posterior, prior).sum(dim=-1)
        weighted_kl_div = kl_weight * kl_div
        # (n_obs, n_vars) -> (n_obs,)
        reconstruction_loss = -likelihood.log_prob(X).sum(dim=-1)

        loss_unreduced = reconstruction_loss + weighted_kl_div
        loss = loss_unreduced.mean()

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kl_local=weighted_kl_div,
            extra_tensors={
                "loss_unreduced": loss_unreduced,
            },
        )
