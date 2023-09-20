import pytest
import torch
from torch import nn

from embedding_scvi._components import MLP, ExtendableEmbedding, MLPBlock, MultiOutputLinear


@pytest.mark.parametrize("n_obs", [10])
@pytest.mark.parametrize("n_in", [10])
@pytest.mark.parametrize("n_out", [10, 20])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("norm", ["batch", "layer", None])
@pytest.mark.parametrize("activation", ["relu", "softmax", None])
@pytest.mark.parametrize("dropout_rate", [0.1, None])
@pytest.mark.parametrize("residual", [True, False])
def test_mlp_block(
    n_obs: int,
    n_in: int,
    n_out: int,
    bias: bool,
    norm: str | None,
    activation: str | None,
    dropout_rate: float,
    residual: bool,
):
    if norm == "batch":
        norm_kwargs = {"eps": 1e-3, "momentum": 0.1}
    elif norm == "layer":
        norm_kwargs = {"eps": 1e-3}
    else:
        norm_kwargs = None

    if activation == "softmax":
        activation_kwargs = {"dim": 1}
    else:
        activation_kwargs = None

    mlp_block = MLPBlock(
        n_in=n_in,
        n_out=n_out,
        bias=bias,
        norm=norm,
        norm_kwargs=norm_kwargs,
        activation=activation,
        activation_kwargs=activation_kwargs,
        dropout_rate=dropout_rate,
        residual=residual,
    )

    x = torch.randn(n_obs, n_in)
    h = mlp_block(x)
    assert h.shape == (n_obs, n_out)


@pytest.mark.parametrize("n_obs", [10])
@pytest.mark.parametrize("n_in", [10])
@pytest.mark.parametrize("n_out", [20])
@pytest.mark.parametrize("n_out_params", [1, 2])
@pytest.mark.parametrize("activation", ["relu", "softmax", None])
def test_multi_output_linear(
    n_obs: int,
    n_in: int,
    n_out: int,
    n_out_params: int,
    activation: str | None,
):
    if activation == "softmax":
        activation_kwargs = {"dim": 1}
    else:
        activation_kwargs = None

    multi_output_linear = MultiOutputLinear(
        n_in=n_in,
        n_out=n_out,
        n_out_params=n_out_params,
        activations=[activation] * n_out_params,
        activation_kwargs=[activation_kwargs] * n_out_params,
    )

    x = torch.randn(n_obs, n_in)
    h = multi_output_linear(x)
    assert len(h) == n_out_params
    assert all(h_i.shape == (n_obs, n_out) for h_i in h)


@pytest.mark.parametrize("n_obs", [10])
@pytest.mark.parametrize("n_in", [10])
@pytest.mark.parametrize("n_out", [20])
@pytest.mark.parametrize("n_hidden", [64])
@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize("bias", [True])
@pytest.mark.parametrize("norm", ["batch"])
@pytest.mark.parametrize("activation", ["relu"])
@pytest.mark.parametrize("dropout_rate", [0.1])
@pytest.mark.parametrize("residual", [True])
def test_mlp(
    n_obs: int,
    n_in: int,
    n_out: int,
    n_hidden: int,
    n_layers: int,
    bias: bool,
    norm: str,
    activation: str,
    dropout_rate: float,
    residual: bool,
):
    if norm == "batch":
        norm_kwargs = {"eps": 1e-3, "momentum": 0.1}
    elif norm == "layer":
        norm_kwargs = {"eps": 1e-3}
    else:
        norm_kwargs = None

    if activation == "softmax":
        activation_kwargs = {"dim": 1}
    else:
        activation_kwargs = None

    mlp = MLP(
        n_in=n_in,
        n_out=n_out,
        n_hidden=n_hidden,
        n_layers=n_layers,
        bias=bias,
        norm=norm,
        norm_kwargs=norm_kwargs,
        activation=activation,
        activation_kwargs=activation_kwargs,
        dropout_rate=dropout_rate,
        residual=residual,
    )

    x = torch.randn(n_obs, n_in)
    h = mlp(x)
    assert h.shape == (n_obs, n_out)


@pytest.mark.parametrize("num_embeddings", [10])
@pytest.mark.parametrize("embedding_dim", [5])
@pytest.mark.parametrize("init", [2, [0, 1]])
@pytest.mark.parametrize("freeze_prev", [True, False])
def test_extendable_embedding(
    num_embeddings: int,
    embedding_dim: int,
    init: int | list[int],
    freeze_prev: bool,
):
    embedding = nn.Embedding(num_embeddings, embedding_dim)
    ext_embedding = ExtendableEmbedding.extend_embedding(embedding, init=init, freeze_prev=freeze_prev)
    n_init = len(init) if isinstance(init, list) else init

    assert ext_embedding.num_embeddings == num_embeddings + n_init
    assert ext_embedding.embedding_dim == embedding_dim
    assert ext_embedding.weight.shape == (num_embeddings + n_init, embedding_dim)
    assert torch.equal(ext_embedding.weight[:num_embeddings], embedding.weight)

    if isinstance(init, list):
        assert torch.equal(ext_embedding.weight[num_embeddings:], embedding.weight[init])

    dummy_indexes = torch.arange(num_embeddings + n_init, dtype=torch.long)
    dummy_prediction = ext_embedding(dummy_indexes)
    dummy_target = torch.randn_like(dummy_prediction)
    dummy_loss = torch.nn.functional.mse_loss(dummy_prediction, dummy_target, reduce=True)
    dummy_loss.backward()
    grad = ext_embedding.weight.grad

    if freeze_prev:
        prev_grad = grad[:num_embeddings]
        new_grad = grad[num_embeddings:]
        assert torch.equal(prev_grad, torch.zeros_like(prev_grad))
        assert not torch.equal(new_grad, torch.zeros_like(new_grad))
    else:
        assert not torch.equal(grad, torch.zeros_like(grad))
