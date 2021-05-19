import logging
import os
import numpy as np
import torch
from torch.nn import functional as F
from result_caching import store_dict
from sklearn.decomposition import PCA
from tqdm import tqdm
from model_tools.activations.core import flatten, change_dict
from model_tools.activations.pca import _get_imagenet_val
from model_tools.utils import fullname


class ImageNetLayerEigenspectrum:

    def __init__(self, activations_extractor, pooling=None):
        assert pooling in [None, 'max', 'avg']
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._pooling = pooling
        self._layer_eigenspectrums = {}

    def effective_dimensionalities(self, layers):
        self._ensure_initialized(layers)
        effdims = {layer: eigspec.sum() ** 2 / (eigspec ** 2).sum()
                   for layer, eigspec in self._layer_eigenspectrums.items()}
        return effdims

    def _ensure_initialized(self, layers):
        missing_layers = [layer for layer in layers if layer not in self._layer_eigenspectrums]
        if len(missing_layers) == 0:
            return
        layer_eigenspectrums = self._eigenspectrums(identifier=self._extractor.identifier,
                                                    layers=missing_layers,
                                                    pooling=self._pooling)
        self._layer_eigenspectrums = {**self._layer_eigenspectrums, **layer_eigenspectrums}

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def _eigenspectrums(self, identifier, layers, pooling):
        self._logger.debug('Retrieving ImageNet activations')
        imagenet_paths = _get_imagenet_val(num_images=1000)
        handle = self._extractor.register_batch_activations_hook(pooling_hook(pooling))
        imagenet_activations = self._extractor(imagenet_paths, layers=layers)
        handle.remove()
        imagenet_activations = {layer: imagenet_activations.sel(layer=layer).values
                                for layer in np.unique(imagenet_activations['layer'])}
        assert len(set(activations.shape[0] for activations in imagenet_activations.values())) == 1, "stimuli differ"

        self._logger.debug('Computing ImageNet principal components')
        progress = tqdm(total=len(imagenet_activations), desc="layer principal components")

        def init_and_progress(layer, activations):
            activations = flatten(activations)
            pca = PCA(random_state=0)
            pca.fit(activations)
            eigenspectrum = pca.explained_variance_
            progress.update(1)
            return eigenspectrum

        from model_tools.activations.core import change_dict
        layer_eigenspectrums = change_dict(imagenet_activations, init_and_progress, keep_name=True,
                                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')
        progress.close()
        return layer_eigenspectrums


def pooling_hook(pooling):
    assert pooling in [None, 'max', 'avg']

    def apply_pooling(layer, activations):
        if pooling is None or activations.ndim != 4:
            return activations
        activations = torch.from_numpy(activations)
        if pooling == 'max':
            activations = F.adaptive_max_pool2d(activations, 1)
        elif pooling == 'avg':
            activations = F.adaptive_avg_pool2d(activations, 1)
        else:
            raise ValueError(f'pooling "{pooling}" is not valid')
        activations = activations.numpy()
        return activations

    def hook(batch_activations):
        return change_dict(batch_activations, apply_pooling, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

    return hook
