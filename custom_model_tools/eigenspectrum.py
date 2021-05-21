import logging
import os
import numpy as np
from result_caching import store_dict
from sklearn.decomposition import PCA
from tqdm import tqdm
from model_tools.activations.core import flatten
from model_tools.activations.pca import _get_imagenet_val
from model_tools.utils import fullname
from custom_model_tools.hooks import GlobalMaxPool2d


class ImageNetLayerEigenspectrum:

    def __init__(self, activations_extractor, pooling=True):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._pooling = pooling
        self._layer_eigenspectra = {}

    def fit(self, layers):
        missing_layers = [layer for layer in layers if layer not in self._layer_eigenspectra]
        if len(missing_layers) == 0:
            return
        layer_eigenspectra = self._fit(identifier=self._extractor.identifier,
                                         layers=missing_layers,
                                         pooling=self._pooling)
        self._layer_eigenspectra = {**self._layer_eigenspectra, **layer_eigenspectra}

    def effective_dimensionalities(self, layers):
        self.fit(layers)
        effdims = {layer: eigspec.sum() ** 2 / (eigspec ** 2).sum()
                   for layer, eigspec in self._layer_eigenspectra.items()}
        return effdims

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def _fit(self, identifier, layers, pooling):
        self._logger.debug('Retrieving ImageNet activations')
        imagenet_paths = _get_imagenet_val(num_images=1000)
        if self._pooling:
            handle = GlobalMaxPool2d.hook(self._extractor)
        imagenet_activations = self._extractor(imagenet_paths, layers=layers)
        if self._pooling:
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
        layer_eigenspectra = change_dict(imagenet_activations, init_and_progress, keep_name=True,
                                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')
        progress.close()
        return layer_eigenspectra
