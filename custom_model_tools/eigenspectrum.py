import logging
from result_caching import store_dict
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from tqdm import tqdm
from model_tools.activations.core import flatten
from model_tools.utils import fullname
from custom_model_tools.hooks import GlobalMaxPool2d
from custom_model_tools.image_transform import ImageDatasetTransformer
from utils import id_to_properties, get_imagenet_val
from typing import Optional


class ImageNetLayerEigenspectrum:

    def __init__(self, activations_extractor, pooling=True,
                 image_transform: Optional[ImageDatasetTransformer] = None):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._pooling = pooling
        self._image_transform = image_transform
        self._layer_eigenspectra = {}

    def fit(self, layers):
        transform_name = None if self._image_transform is None else self._image_transform.name
        self._layer_eigenspectra = self._fit(identifier=self._extractor.identifier,
                                             layers=layers,
                                             pooling=self._pooling,
                                             image_transform_name=transform_name)

    def effective_dimensionalities(self):
        effdims = {layer: eigspec.sum() ** 2 / (eigspec ** 2).sum()
                   for layer, eigspec in self._layer_eigenspectra.items()}
        return effdims

    def eighty_percent_var(self):
        eighty_percent_var = {}
        for layer, eigspec in self._layer_eigenspectra.items():
            pvar = eigspec.cumsum() / eigspec.sum()
            for i in range(len(pvar)):
                if pvar[i] >= 0.8:
                    eighty_percent_var[layer] = i + 1
                    break
        return eighty_percent_var

    def powerlaw_exponent(self):
        alpha = {}
        for layer, eigspec in self._layer_eigenspectra.items():
            start = 0
            end = np.log10(len(eigspec))
            eignum = np.logspace(start, end, num=50).round().astype(int)
            eigspec = eigspec[eignum - 1]
            logeignum = np.log10(eignum)
            logeigspec = np.log10(eigspec)
            linear_fit = LinearRegression().fit(logeignum.reshape(-1, 1), logeigspec)
            alpha[layer] = -linear_fit.coef_.item()
        return alpha

    def as_df(self):
        df = pd.DataFrame()
        for layer, eigspec in self._layer_eigenspectra.items():
            layer_df = pd.DataFrame({'n': range(1, len(eigspec) + 1), 'variance': eigspec})
            layer_df = layer_df.assign(layer=layer)
            df = df.append(layer_df)
        properties = id_to_properties(self._extractor.identifier)
        df = df.assign(**properties)
        return df

    def metrics_as_df(self):
        effdims = self.effective_dimensionalities()
        eightyvar = self.eighty_percent_var()
        alpha = self.powerlaw_exponent()
        df = pd.DataFrame()
        for layer in self._layer_eigenspectra:
            df = df.append({'layer': layer,
                            'effective dimensionality': effdims[layer],
                            '80% variance': eightyvar[layer],
                            'alpha': alpha[layer]},
                           ignore_index=True)
        properties = id_to_properties(self._extractor.identifier)
        df = df.assign(**properties)
        return df

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def _fit(self, identifier, layers, pooling, image_transform_name):
        imagenet_paths = get_imagenet_val(num_per_class=10)
        if self._image_transform is not None:
            imagenet_paths = self._image_transform.transform_dataset('imagenetval', imagenet_paths)
        if self._pooling:
            handle = GlobalMaxPool2d.hook(self._extractor)

        # Compute activations and PCA for every layer individually to save on memory.
        # This is more inefficient because we run images through the network several times,
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
