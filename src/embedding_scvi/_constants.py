from typing import NamedTuple


class _TENSORS_KEYS_NT(NamedTuple):
    Z_KEY: str = "z"
    QZ_KEY: str = "qz"
    PZ_KEY: str = "pz"
    PX_KEY: str = "px"
    COVARIATES_Z_KEY: str = "covariates_z"
    LIBRARY_SIZE_KEY: str = "library_size"


TENSORS_KEYS = _TENSORS_KEYS_NT()
