from os import listdir
from os.path import isfile, join, dirname, abspath, exists
import urllib.request
import tempfile
import shutil


def ensure_demo_data():
    """Check if demo data is present. Else download from github."""
    # Very basic check for demo data. Delete the folder and re-run if data is
    # corrupted.
    DEMO_DATA_DIR = join(dirname(abspath(__file__)), "demo_data")
    DEMO_DATA_URL = "https://github.com/isl-org/open3d_downloads/releases/download/open3d-ml/open3dml_demo_data.zip"
    if exists(DEMO_DATA_DIR) and {'KITTI', 'SemanticKITTI'}.issubset(
            listdir(DEMO_DATA_DIR)):
        return DEMO_DATA_DIR
    print(f"Demo data not found in {DEMO_DATA_DIR}. Downloading...")
    with tempfile.TemporaryDirectory() as dl_dir:
        dl_filename = join(dl_dir, "demo_data.zip")
        with urllib.request.urlopen(DEMO_DATA_URL) as response, open(
                dl_filename, 'wb') as dl_file:
            shutil.copyfileobj(response, dl_file)
        shutil.unpack_archive(dl_filename, DEMO_DATA_DIR)

    return DEMO_DATA_DIR
