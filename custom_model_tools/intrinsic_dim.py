from __future__ import annotations
import logging
import os.path
from result_caching import store_dict
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
import pandas as pd
import numpy as np
from tqdm import tqdm
from brainio_base.assemblies import NeuroidAssembly
from model_tools.activations.core import flatten
from model_tools.utils import fullname
from custom_model_tools.hooks import GlobalMaxPool2d, RandomProjection
from utils import id_to_properties, get_imagenet_val
from typing import List, Tuple


class IntrinsicDimBase:

    def __init__(self, activations_extractor, pooling=True, stimuli_identifier=None):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._pooling = pooling
        self._stimuli_identifier = stimuli_identifier
        self._layer_dims = {}

    def fit(self, layers):
        self._layer_dims = self._fit(identifier=self._extractor.identifier,
                                     stimuli_identifier=self._stimuli_identifier,
                                     layers=layers,
                                     pooling=self._pooling)

    def as_df(self):
        df = pd.DataFrame()
        for layer, statistics in self._layer_dims.items():
            statistics['layer'] = layer
            df = df.append(statistics, ignore_index=True)
        properties = id_to_properties(self._extractor.identifier)
        df = df.assign(**properties)
        return df

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def _fit(self, identifier, stimuli_identifier, layers, pooling):
        concept_paths = self.get_image_concept_paths()

        # Compute manifold geometry statistics for every layer individually to save on memory.
        # This is more inefficient because we run images through the network several times,
        # but it is a more scalable approach when using many images and large layers.
        layer_manifold_statistics = {}
        for layer in layers:
            if pooling:
                handle = GlobalMaxPool2d.hook(self._extractor)
            else:
                handle = RandomProjection.hook(self._extractor)

            self._logger.debug('Computing concept manifold geometries')
            concept_manifolds = []
            for stimuli_paths in tqdm(concept_paths, desc='concept manifolds'):
                activations = self._extractor(stimuli_paths, layers=[layer])
                activations = activations.sel(layer=layer).values
                activations = flatten(activations)
                concept_manifolds.append(ManifoldGeometry(activations))

            self._logger.debug('Computing concept manifold statistics')
            progress = tqdm(total=1, desc="manifold statistics")
            layer_manifold_statistics[layer] = get_manifold_statistics(concept_manifolds)
            progress.update(1)
            progress.close()

            handle.remove()

        return layer_manifold_statistics

    def get_image_concept_paths(self) -> List[List[str]]:
        raise NotImplementedError()
