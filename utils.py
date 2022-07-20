import logging
import os
from time import time
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import h5py
from PIL import Image
from model_tools.utils import fullname, s3


def properties_to_id(architecture, task, kind, source):
    identifier = f'architecture:{architecture}|task:{task}|kind:{kind}|source:{source}'
    return identifier


def id_to_properties(identifier):
    identifier = identifier.split(',')[0]
    properties = identifier.split('|')[:4]
    properties = {p.split(':')[0]: p.split(':')[1] for p in properties}
    return properties


def timed(func):
    """Decorator that reports the execution time."""
    def wrap(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        elapsed = str(datetime.timedelta(seconds=(end - start)))
        print(f'{func.__name__} total runtime: {elapsed}')
        return result
    return wrap


def get_imagenet_val(num_classes=1000, num_per_class=1, separate_classes=False):
    _logger = logging.getLogger(fullname(get_imagenet_val))
    base_indices = np.arange(num_per_class).astype(int)
    indices = []
    for i in range(num_classes):
        indices.extend(50 * i + base_indices)

    framework_home = os.path.expanduser(os.getenv('MT_HOME', '~/.model-tools'))
    imagenet_filepath = os.getenv('MT_IMAGENET_PATH', os.path.join(framework_home, 'imagenet2012.hdf5'))
    imagenet_dir = f"{imagenet_filepath}-files"
    os.makedirs(imagenet_dir, exist_ok=True)

    if not os.path.isfile(imagenet_filepath):
        os.makedirs(os.path.dirname(imagenet_filepath), exist_ok=True)
        _logger.debug(f"Downloading ImageNet validation to {imagenet_filepath}")
        s3.download_file("imagenet2012-val.hdf5", imagenet_filepath)

    filepaths = []
    with h5py.File(imagenet_filepath, 'r') as f:
        for index in indices:
            imagepath = os.path.join(imagenet_dir, f"{index}.png")
            if not os.path.isfile(imagepath):
                image = np.array(f['val/images'][index])
                Image.fromarray(image).save(imagepath)
            filepaths.append(imagepath)

    if separate_classes:
        filepaths = [filepaths[i * num_per_class:(i + 1) * num_per_class]
                     for i in range(num_classes)]

    return filepaths


def fix_alpha(eigspectra_path, eigmetrics_path):
    id_columns = ['architecture', 'task', 'kind', 'source', 'layer']
    eigspectra, eigmetrics = pd.read_csv(eigspectra_path), pd.read_csv(eigmetrics_path)

    def is_outlier(n):
        n = np.log(n)
        cutoff = 0.95 * n.iloc[-1]
        return n > cutoff
    outlier = eigspectra.groupby(id_columns)['n'].transform(is_outlier)
    eigspectra = eigspectra[~outlier]

    def get_alpha(x):
        n, y = x['n'].values, x['variance'].values
        n, y = np.log(n), np.log(y)
        coef = LinearRegression().fit(n.reshape(-1, 1), y).coef_[0]
        alpha = -coef
        return alpha
    alphas = eigspectra.groupby(id_columns).apply(get_alpha)

    eigmetrics = eigmetrics.set_index(id_columns)
    eigmetrics['alpha'] = alphas
    eigmetrics = eigmetrics.reset_index(id_columns)

    eigmetrics.to_csv(eigmetrics_path, index=False)
