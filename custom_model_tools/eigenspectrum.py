import logging
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
        imagenet_paths = _get_imagenet_val(num_images=10000)
        if self._pooling:
            handle = GlobalMaxPool2d.hook(self._extractor)

        # Compute activations and PCA for every layer individually to save on memory.
        # This is more inefficient because we we'run images through the network several times,
        # but it is a more scalable approach when using many images and large layers.
        layer_eigenspectra = {}
        for layer in layers:
            self._logger.debug('Retrieving ImageNet activations')
            activations = self._extractor(imagenet_paths, layers=[layer])
            activations = activations.sel(layer=layer).values

            self._logger.debug('Computing ImageNet principal components')
            progress = tqdm(total=1, desc="layer principal components")
            activations = flatten(activations)
            pca = PCA(random_state=0)
            pca.fit(activations)
            eigenspectrum = pca.explained_variance_
            progress.update(1)
            progress.close()

            layer_eigenspectra[layer] = eigenspectrum

        if self._pooling:
            handle.remove()

        return layer_eigenspectra
