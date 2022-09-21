from src.paths import MODEL_PATHS as PATHS  # Used in __all__ and submodules.
from .chamfer_distance import *
from .databases import *

__all__ = (
        chamfer_distance.__all__ +
        databases.__all__ +
        ['PATHS']
)
