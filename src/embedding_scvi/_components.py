import numpy as np
import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        bias: bool = True,
        norm: str | None = None,
        activation: str | None = None,
        dropout_rate: float | None = None,
        residual: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out, bias=bias)
        self.residual = residual

        if norm == "batch":
            self.norm = nn.BatchNorm1d(n_out)
        elif norm == "layer":
            self.norm = nn.LayerNorm(n_out)
        elif norm is not None:
            raise ValueError(f"Unrecognized norm: {norm}")
        else:
            self.norm = nn.Identity()

        if activation is not None:
            self.activation = getattr(nn.functional, activation)
        else:
            self.activation = nn.Identity()

        if dropout_rate is not None:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()

        if self.residual and n_in != n_out:
            raise ValueError("`n_in` must equal `n_out` if `residual` is `True`.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = self.norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = h + x if self.residual else h
        return h


class MultiOutput(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_out_params: int,
        param_activations: list[int] | None,
    ):
        super().__init__()
        self.n_out_params = n_out_params
        self.param_activations = param_activations

        if self.param_activations is not None and len(param_activations) != n_out_params:
            raise ValueError(
                f"Length of `param_activations` {len(param_activations)}) must "
                f"match `n_out_params`: {n_out_params}."
            )
        elif self.param_activations is None:
            self.param_activations = [None for _ in range(n_out_params)]

        blocks = []
        for i in range(self.n_out_params):
            blocks.append(
                MLPBlock(
                    n_in=n_in,
                    n_out=n_out,
                    bias=False,
                    activation=self.param_activations[i],
                )
            )
        self._blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(block(x) for block in self._blocks)


class MLP(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int,
        n_layers: int,
        bias: bool = True,
        norm: str | None = None,
        activation: str | None = None,
        dropout_rate: float | None = None,
        residual: bool = False,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        n_ins = [n_in] + [n_hidden for _ in range(n_layers - 1)]
        n_outs = [n_hidden for _ in range(n_layers - 1)] + [n_out]
        blocks = []
        for n_in, n_out in zip(n_ins, n_outs):
            _residual = residual and n_in == n_out
            blocks.append(
                MLPBlock(
                    n_in=n_in,
                    n_out=n_out,
                    bias=bias,
                    norm=norm,
                    activation=activation,
                    dropout_rate=dropout_rate,
                    residual=_residual,
                )
            )

        self._blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._blocks(x)


class MLPMultiOutput(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int,
        n_layers: int,
        n_out_params: int,
        param_activations: list[int] | None,
        bias: bool = True,
        norm: str | None = None,
        activation: str | None = None,
        dropout_rate: float | None = None,
        residual: bool = False,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self._mlp = MLP(
            n_in=n_in,
            n_out=n_hidden,
            n_hidden=n_hidden,
            n_layers=n_layers,
            bias=bias,
            norm=norm,
            activation=activation,
            dropout_rate=dropout_rate,
            residual=residual,
        )
        self._multi_output = MultiOutput(
            n_in=n_hidden,
            n_out=n_out,
            n_out_params=n_out_params,
            param_activations=param_activations,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self._mlp(x)
        return self._multi_output(h)


class ExtendableEmbedding(nn.Embedding):
    @classmethod
    def extend_embedding(
        cls,
        embedding: nn.Embedding,
        init: int | list[int],
        freeze_prev: bool = True,
    ):
        old_weight = embedding.weight.clone()
        if isinstance(init, int) and init > 0:
            num_init = init
            new_weight = torch.empty(
                (init, old_weight.shape[1]),
                device=old_weight.device,
            )
            nn.init.normal_(new_weight)
        elif isinstance(init, list):
            num_init = len(init)
            new_weight = old_weight[init]
        weight = torch.cat([old_weight, new_weight], dim=0)

        new_embedding = cls(
            num_embeddings=embedding.num_embeddings + num_init,
            embedding_dim=embedding.embedding_dim,
            _weight=weight,
            padding_idx=embedding.padding_idx,
            max_norm=embedding.max_norm,
            norm_type=embedding.norm_type,
            scale_grad_by_freq=embedding.scale_grad_by_freq,
            sparse=embedding.sparse,
        )

        def _partial_freeze_hook(grad: torch.Tensor) -> torch.Tensor:
            grad[: old_weight.shape[0]] = 0
            return grad

        if freeze_prev:
            new_embedding.weight.register_hook(_partial_freeze_hook)

        return new_embedding

    def _load_from_state_dict(self, state_dict, *args, **kwargs):
        for key, val in state_dict.items():
            if "weight" not in key:
                continue
            self.weight = nn.Parameter(val)
            self.num_embeddings = val.shape[0]
            self.embedding_dim = val.shape[1]
            break

        return super()._load_from_state_dict(state_dict, *args, **kwargs)


class ExtendableEmbeddingList(nn.Module):
    def __init__(
        self,
        num_embeddings: list[int],
        **kwargs,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings

        self._embeddings = nn.ModuleList(
            [
                ExtendableEmbedding(
                    num_embeddings=c,
                    **kwargs,
                )
                for c in num_embeddings
            ]
        )

    def forward(self, x: torch.Tensor, subset: int | list[int] | None = None) -> torch.Tensor:
        if isinstance(subset, int):
            subset = [subset]
        elif subset is None:
            subset = list(range(len(self._embeddings)))

        embeddings_subset = [self._embeddings[i] for i in subset]

        return torch.cat(
            [embedding(x[:, i]).unsqueeze(0) for i, embedding in enumerate(embeddings_subset)],
        )

    def get_embedding_layer(self, index: int) -> nn.Embedding:
        return self._embeddings[index]

    def set_embedding_layer(self, index: int, embedding: nn.Embedding):
        self._embeddings[index] = embedding

    def get_embedding_weight(self, index: int, as_tensor: bool = False) -> np.ndarray | torch.Tensor:
        weight = self._embeddings[index].weight.detach().cpu()
        if as_tensor:
            return weight
        return weight.numpy()
