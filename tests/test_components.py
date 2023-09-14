import torch

from embedding_scvi._components import MLP


def test_mlp():
    x = torch.randn(100, 10)
    mlp = MLP(
        n_in=10,
        n_out=10,
        n_out_params=10,
        n_hidden=10,
        n_layers=2,
        bias=True,
        norm=None,
        norm_kwargs=None,
        activation=None,
        activation_kwargs=None,
        dropout_rate=None,
        residual=False,
    )
    y = mlp(x)
    print(len(y))
