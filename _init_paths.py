import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
dateLoader_path = osp.join(this_dir, 'dataLoaders')
add_path(dateLoader_path)

models_path = osp.join(this_dir, 'models')
add_path(models_path)

utils_path = osp.join(this_dir, 'tools')
add_path(utils_path)

sys.path
