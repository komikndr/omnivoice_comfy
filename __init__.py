"""Top-level package for omnivoice_comfy."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

__version__ = "0.0.3"

from .src.omnivoice_comfy import runtime
from .src.omnivoice_comfy.nodes import NODE_CLASS_MAPPINGS
from .src.omnivoice_comfy.nodes import NODE_DISPLAY_NAME_MAPPINGS

runtime.register_model_path()
