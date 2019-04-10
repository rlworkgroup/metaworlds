import os
import pathlib

from dm_control.utils import io as resources

_PALLET_DIR = pathlib.Path(
    os.environ.get('METAWORLDS_PALLET_DIR')).expanduser()

_PALLET_ASSET_FILENAMES = [
    "./utils/basic_scene.xml",
]

ASSETS = {
    filename: resources.GetResource(_PALLET_DIR / filename)
    for filename in _PALLET_ASSET_FILENAMES
}


def read_model(model_filename):
    """Reads a model XML file and returns its contents as a string."""
    return resources.GetResource(os.path.join(_PALLET_DIR, model_filename))
