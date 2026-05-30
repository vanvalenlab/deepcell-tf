"""Deepcell Utilities Module"""


# compute_overlap has been moved to deepcell_toolbox
# leaving here for backwards compatibility
from deepcell_toolbox import compute_overlap

from deepcell.utils import backbone_utils
from deepcell.utils import data_utils
from deepcell.utils import io_utils
from deepcell.utils import misc_utils
from deepcell.utils import plot_utils
from deepcell.utils import tracking_utils
from deepcell.utils import train_utils
from deepcell.utils import transform_utils
from deepcell.utils import tfrecord_utils

# Globally-importable utils.
from deepcell.utils.data_utils import get_data
from ._auth import fetch_data, extract_archive
from deepcell.utils.misc_utils import sorted_nicely
from deepcell.utils.train_utils import rate_scheduler
from deepcell.utils.transform_utils import outer_distance_transform_2d
from deepcell.utils.transform_utils import outer_distance_transform_3d
from deepcell.utils.transform_utils import outer_distance_transform_movie
from deepcell.utils.transform_utils import inner_distance_transform_2d
from deepcell.utils.transform_utils import inner_distance_transform_3d
from deepcell.utils.transform_utils import inner_distance_transform_movie
from deepcell.utils.transform_utils import pixelwise_transform


def __getattr__(name):
    if name in {"export_model", "export_utils"}:
        import warnings
        import importlib

        warnings.warn(
            f"\n\n{name} is deprecated, use tf.keras.models.save_model directly.\n",
            DeprecationWarning,
            stacklevel=2,
        )

        export_utils = importlib.import_module("deepcell.utils.export_utils")
        if name == "export_model":
            return getattr(export_utils, name)
        if name == "export_utils":
            return export_utils
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
