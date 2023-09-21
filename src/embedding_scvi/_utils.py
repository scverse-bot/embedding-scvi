from __future__ import annotations

from scvi.utils._exceptions import InvalidParameterError


def likelihood_to_dist_params(likelihood: str) -> dict[str, str | None]:
    if likelihood == "normal":
        return {
            "loc": None,
            "scale": "softplus",
        }
    elif likelihood == "zinb":
        return {
            "scale": "softplus",
            "r": None,
            "dropout": None,
        }
    elif likelihood == "nb":
        return {
            "mu": None,
            "theta": "softplus",
            "scale": None,
        }
    elif likelihood == "poisson":
        return {
            "mu": None,
            "scale": None,
        }
    else:
        raise InvalidParameterError(
            param="likelihood",
            value=likelihood,
            valid=["normal", "zinb", "nb", "poisson"],
        )
