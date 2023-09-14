from importlib.metadata import version

from ._model import EmbeddingSCVI

__version__ = version("embedding-scvi")

__all__ = ["EmbeddingSCVI"]
