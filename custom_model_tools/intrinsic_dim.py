from __future__ import annotations
import logging
import os.path
from result_caching import store_dict
import pandas as pd
import numpy as np
from brainio_base.assemblies import NeuroidAssembly
from model_tools.activations.core import flatten
from model_tools.utils import fullname
from custom_model_tools.hooks import GlobalMaxPool2d, RandomProjection
from lib.twoNN import TwoNearestNeighbors
from utils import id_to_properties, get_imagenet_val
from typing import List, Union


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
        image_paths = self.get_image_paths()
        if isinstance(image_paths[0], list):
            cat_sizes = [0] + [len(cat) for cat in image_paths]
            cat_sizes = np.cumsum(cat_sizes)
            image_paths = [img for cat in image_paths for img in cat]
        else:
            cat_sizes = None

        # Compute intrinsic dimensionality for every layer individually to save on memory.
        # This is more inefficient because we run images through the network several times,
        # but it is a more scalable approach when using many images and large layers.
        layer_dims = {}
        two_nn = TwoNearestNeighbors()
        for layer in layers:
            if pooling:
                handle = GlobalMaxPool2d.hook(self._extractor)
            else:
                handle = RandomProjection.hook(self._extractor)

            self._logger.debug('Retrieving activations')
            activations = self._extractor(image_paths, layers=[layer])
            activations = activations.sel(layer=layer).values
            activations = flatten(activations)

            self._logger.debug('Computing ID')
            dim_global = two_nn.fit(activations).dim_

            if cat_sizes is not None:
                self._logger.debug('Computing inter/intra-class ID')
                cat_means = []
                dims_intra = []
                for i in range(len(cat_sizes) - 1):
                    cat_activations = activations[cat_sizes[i]:cat_sizes[i + 1]]
                    cat_means.append(cat_activations.mean(axis=0))
                    cat_dim_intra = two_nn.fit(cat_activations).dim_
                    dims_intra.append(cat_dim_intra)
                dims_intra = np.stack(dims_intra)
                dim_intra_mean, dim_intra_std = dims_intra.mean(axis=0), dims_intra.std(axis=0)
                dim_inter = two_nn.fit(np.stack(cat_means)).dim_

            if cat_sizes is None:
                layer_dims[layer] = {'ID': dim_global}
            else:
                layer_dims[layer] = {'ID': dim_global, 'inter-ID': dim_inter, 'intra-ID': dim_intra_mean,
                                     'intra-ID (std)': dim_intra_std}

            handle.remove()

        return layer_dims

    def get_image_paths(self) -> Union[List[str], List[List[str]]]:
        raise NotImplementedError()


class IntrinsicDimImageNet(IntrinsicDimBase):

    def __init__(self, *args, num_classes=50, num_per_class=50, **kwargs):
        super().__init__(*args, **kwargs, 
                         stimuli_identifier='imagenet')
        assert 2 <= num_classes <= 1000 and 2 <= num_per_class <= 50
        self.num_classes = num_classes
        self.num_per_class = num_per_class
        self.image_paths = get_imagenet_val(num_classes, num_per_class, separate_classes=True)

    def get_image_paths(self) -> List[List[str]]:
        return self.image_paths


class IntrinsicDimImageFolder(IntrinsicDimBase):

    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert os.path.exists(data_dir)
        self.data_dir = data_dir

        image_paths = []
        categories = os.listdir(self.data_dir)
        for cat in categories:
            cat_dir = os.path.join(self.data_dir, cat)
            files = os.listdir(cat_dir)
            paths = [os.path.join(cat_dir, file) for file in files]
            image_paths.append(paths)
        self.image_paths = image_paths

    def get_image_paths(self) -> List[List[str]]:
        return self.image_paths


class IntrinsicDimImageNet21k(IntrinsicDimImageFolder):

    def __init__(self, *args, num_classes=50, num_per_class=50, **kwargs):
        super().__init__(*args, **kwargs, 
                         stimuli_identifier='imagenet21k')
        self.num_classes = num_classes
        self.num_per_class = num_per_class

        assert len(self.image_paths) >= num_classes
        self.image_paths = self.image_paths[:num_classes]
        for i in range(len(self.image_paths)):
            assert len(self.image_paths[i]) >= num_per_class
            self.image_paths[i] = self.image_paths[i][:num_classes]


class IntrinsicDimObject2Vec(IntrinsicDimImageFolder):

    def __init__(self, data_dir, *args, **kwargs):
        data_dir = os.path.join(data_dir, 'stimuli_rgb')
        super().__init__(data_dir, *args, **kwargs,
                         stimuli_identifier='object2vec')


class IntrinsicDimMajajHong2015(IntrinsicDimBase):
    # Brainscore IT benchmark images (64 objects, 50 images/object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         stimuli_identifier='dicarlo.hvm-public')

        data_dir = os.getenv('BRAINIO_HOME', os.path.expanduser('~/.brainio'))
        data_dir = os.path.join(data_dir, 'image_dicarlo_hvm-public')
        assert os.path.exists(data_dir)

        image_paths = pd.read_csv(os.path.join(data_dir, 'image_dicarlo_hvm-public.csv'))
        image_paths = image_paths[['object_name', 'filename']]
        image_paths['filename'] = data_dir + '/' + image_paths['filename']
        image_paths = image_paths.groupby('object_name')['filename'].agg(list)
        image_paths = image_paths.values.tolist()
        self.image_paths = image_paths

    def get_image_image_paths(self) -> List[List[str]]:
        return self.image_paths


def neural_assembly_intrinsic_dim(assembly: NeuroidAssembly, cat_coord: str):
    two_nn = TwoNearestNeighbors()

    dim_global = two_nn.fit(assembly.values).dim_

    activations = [g[1].values for g in assembly.groupby(cat_coord)]
    cat_means = []
    dims_intra = []
    for cat_activations in activations:
        cat_means.append(cat_activations.mean(axis=0))
        cat_dim_intra = two_nn.fit(cat_activations).dim_
        dims_intra.append(cat_dim_intra)
    dims_intra = np.stack(dims_intra)
    dim_intra_mean, dim_intra_std = dims_intra.mean(axis=0), dims_intra.std(axis=0)
    dim_inter = two_nn.fit(np.stack(cat_means)).dim_

    return {'ID': dim_global, 'inter-ID': dim_inter, 'intra-ID': dim_intra_mean,
            'intra-ID (std)': dim_intra_std}
