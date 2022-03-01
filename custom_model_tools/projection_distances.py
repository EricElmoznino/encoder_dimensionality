from __future__ import annotations
import itertools
import logging
import os
import numpy as np
import xarray as xr
from result_caching import store_xarray
from model_tools.activations.core import flatten
from model_tools.utils import fullname
from custom_model_tools.hooks import GlobalMaxPool2d, RandomProjection
from utils import id_to_properties, get_imagenet_val
from typing import List, Dict


class ProjectionDistancesBase:

    def __init__(self, activations_extractor, pooling=True, stimuli_identifier=None):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._pooling = pooling
        self._stimuli_identifier = stimuli_identifier
        self._layer_projection_distances = None

    def fit(self, layers):
        self._layer_projection_distances = self._fit(identifier=self._extractor.identifier,
                                                     stimuli_identifier=self._stimuli_identifier,
                                                     layers=layers,
                                                     pooling=self._pooling)

    def as_dataarray(self):
        assert self._layer_projection_distances is not None
        da = self._layer_projection_distances
        properties = id_to_properties(self._extractor.identifier)
        property_coords = {prop: ('layer', [val] * da.sizes['layer'])
                           for prop, val in properties.items()}
        da = da.assign_coords(property_coords)
        da = da.set_index(identifier=[prop for prop in properties] + ['layer'])
        return da

    @store_xarray(identifier_ignore=['layers'], combine_fields={'layers': 'layer'})
    def _fit(self, identifier, stimuli_identifier, layers, pooling):
        cat_paths = self.get_image_paths()
        cat_names = list(cat_paths.keys())
        num_cats = len(cat_paths)
        assert num_cats > 0
        num_samples = len(list(cat_paths.values())[0])
        for paths in cat_paths.values():
            assert len(paths) == num_samples
        flattened_paths = [path for cat in cat_paths.values() for path in cat]

        # Compute sample projections along class reading directions for every layer individually to save on memory.
        # This is more inefficient because we run images through the network several times,
        # but it is a more scalable approach when using many images and large layers.
        layer_projection_distances = xr.DataArray(data=np.zeros((2, len(cat_paths), len(cat_paths), len(layers))),
                                                  dims=['metric', 'source_category', 'target_category', 'layer'],
                                                  coords={
                                                      'metric': ['centroid_distances', 'scaled_projections'],
                                                      'source_category': cat_names,
                                                      'target_category': cat_names,
                                                      'layer': layers})
        for layer in layers:
            if pooling:
                handle = GlobalMaxPool2d.hook(self._extractor)
            else:
                handle = RandomProjection.hook(self._extractor)

            # Get all activations
            self._logger.debug('Retrieving activations')
            activations = self._extractor(flattened_paths, layers=[layer])
            activations = activations.sel(layer=layer).values
            activations = flatten(activations)
            activations = activations.reshape(num_cats, num_samples, -1)

            self._logger.debug('Computing pairwise class sample projection distances')
            cat_centroids = activations.mean(axis=1)
            for i, j in itertools.product(range(num_cats), range(num_cats)):
                if i == j:
                    continue

                # Compute inter-class centroid distance
                cat_i_centroid, cat_j_centroid = cat_centroids[i], cat_centroids[j]
                cat_i_to_j = cat_j_centroid - cat_i_centroid
                cat_i_to_j_distance = np.linalg.norm(cat_i_to_j)

                # Compute class sample projections along readout direction
                cat_i_samples = activations[i]
                cat_i_samples = cat_i_samples - cat_i_centroid
                cat_i_to_j_projections = np.abs(cat_i_samples @ cat_i_to_j) / cat_i_to_j_distance

                # Scale projections according to manifold radius (sqrt of average dimension variance)
                radius = np.sqrt(cat_i_samples.var(axis=0, ddof=1).mean())
                cat_i_to_j_projections_scaled = cat_i_to_j_projections / radius

                layer_projection_distances.loc[{'layer': layer,
                                                'source_category': cat_names[i],
                                                'target_category': cat_names[j]}] = \
                    [cat_i_to_j_distance, cat_i_to_j_projections_scaled.mean()]

            handle.remove()

        return layer_projection_distances

    def get_image_paths(self) -> Dict[str, List[str]]:
        raise NotImplementedError()


class ProjectionDistancesImageNet(ProjectionDistancesBase):

    def __init__(self, *args, n_cats=50, n_samples=50, **kwargs):
        super(ProjectionDistancesImageNet, self).__init__(*args, **kwargs, stimuli_identifier='imagenet')
        self._n_cats = n_cats
        self._n_samples = n_samples

    def get_image_paths(self) -> Dict[str, List[str]]:
        cat_paths = get_imagenet_val(num_classes=self._n_cats,
                                     num_per_class=self._n_samples,
                                     separate_classes=True)
        cat_paths = {str(i): paths for i, paths in enumerate(cat_paths)}
        return cat_paths


class ProjectionDistancesImageFolder(ProjectionDistancesBase):

    def __init__(self, data_dir, *args, n_cats=50, n_samples=50, **kwargs):
        super(ProjectionDistancesImageFolder, self).__init__(*args, **kwargs)

        assert os.path.exists(data_dir)
        self.data_dir = data_dir
        self._n_cats = n_cats
        self._n_samples = n_samples

        cat_paths = {}
        cats = os.listdir(data_dir)
        assert len(cats) >= n_cats
        cats = cats[:n_cats]
        for cat in cats:
            cat_dir = os.path.join(data_dir, cat)
            files = os.listdir(cat_dir)
            assert len(files) >= n_samples
            files = files[:n_samples]
            paths = [os.path.join(cat_dir, file) for file in files]
            cat_paths[cat] = paths
        self._cat_paths = cat_paths

    def get_image_paths(self) -> Dict[str, List[str]]:
        return self._cat_paths


class ProjectionDistancesObject2Vec(ProjectionDistancesImageFolder):

    def __init__(self, data_dir, *args, **kwargs):
        data_dir = os.path.join(data_dir, 'stimuli_rgb')
        super(ProjectionDistancesObject2Vec, self).__init__(data_dir, *args, **kwargs,
                                                            n_cats=81, n_samples=10,
                                                            stimuli_identifier='object2vec')
