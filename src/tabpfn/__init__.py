from importlib.metadata import version

from tabpfn.classifier import TabPFNClassifier
from tabpfn.misc.debug_versions import display_debug_info
from tabpfn.regressor import TabPFNRegressor

from tabpfn.encoder import TabPFNEncoder

try:
    __version__ = version(__name__)
except ImportError:
    __version__ = "unknown"

__all__ = [
    "TabPFNClassifier",
    "TabPFNRegressor",
    "TabPFNEncoder",
    "__version__",
    "display_debug_info",
]
