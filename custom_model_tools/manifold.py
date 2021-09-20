from __future__ import annotations
import logging
import os.path
from result_caching import store_dict
import pandas as pd
from tqdm import tqdm
from brainio_base.assemblies import NeuroidAssembly
from model_tools.activations.core import flatten
from model_tools.utils import fullname
from custom_model_tools.hooks import GlobalMaxPool2d, RandomProjection
from lib.manifold_geometry import ManifoldGeometry, get_manifold_statistics
from utils import id_to_properties, get_imagenet_val
from typing import List


class ManifoldStatisticsBase:

    def __init__(self, activations_extractor, pooling=True, stimuli_identifier=None):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._pooling = pooling
        self._stimuli_identifier = stimuli_identifier
        self._layer_manifold_statistics = {}

    def fit(self, layers):
        self._layer_manifold_statistics = self._fit(identifier=self._extractor.identifier,
                                                    stimuli_identifier=self._stimuli_identifier,
                                                    layers=layers,
                                                    pooling=self._pooling)

    def as_df(self):
        df = pd.DataFrame()
        for layer, statistics in self._layer_manifold_statistics.items():
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


class ManifoldStatisticsImageNet(ManifoldStatisticsBase):

    def __init__(self, activations_extractor, num_classes=50, num_per_class=50, pooling=True):
        super().__init__(activations_extractor, pooling, 'imagenet')
        assert 2 <= num_classes <= 1000 and 2 <= num_per_class <= 50
        self.num_classes = num_classes
        self.num_per_class = num_per_class
        self.concept_paths = get_imagenet_val(num_classes, num_per_class, separate_classes=True)

    def get_image_concept_paths(self) -> List[List[str]]:
        return self.concept_paths


class ManifoldStatisticsImageFolder(ManifoldStatisticsBase):

    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert os.path.exists(data_dir)
        self.data_dir = data_dir

        concept_paths = []
        concepts = os.listdir(self.data_dir)
        for concept in concepts:
            concept_dir = os.path.join(self.data_dir, concept)
            files = os.listdir(concept_dir)
            paths = [os.path.join(concept_dir, file) for file in files]
            concept_paths.append(paths)
        self.concept_paths = concept_paths

    def get_image_concept_paths(self) -> List[List[str]]:
        return self.concept_paths


class ManifoldStatisticsImageNet21k(ManifoldStatisticsImageFolder):

    def __init__(self, data_dir, num_classes=50, num_per_class=50, *args, **kwargs):
        super().__init__(data_dir, *args, **kwargs,
                         stimuli_identifier='imagenet21k')
        self.num_classes = num_classes
        self.num_per_class = num_per_class

        assert len(self.concept_paths) >= num_classes
        self.concept_paths = self.concept_paths[:num_classes]
        for i in range(len(self.concept_paths)):
            assert len(self.concept_paths[i]) >= num_per_class
            self.concept_paths[i] = self.concept_paths[i][:num_classes]


class ManifoldStatisticsObject2Vec(ManifoldStatisticsImageFolder):

    def __init__(self, data_dir, *args, **kwargs):
        data_dir = os.path.join(data_dir, 'stimuli_rgb')
        super().__init__(data_dir, *args, **kwargs,
                         stimuli_identifier='object2vec')


class ManifoldStatisticsMajajHong2015(ManifoldStatisticsBase):
    # Brainscore IT benchmark images (64 objects, 50 images/object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         stimuli_identifier='dicarlo.hvm-public')

        data_dir = os.getenv('BRAINIO_HOME', os.path.expanduser('~/.brainio'))
        data_dir = os.path.join(data_dir, 'image_dicarlo_hvm-public')
        assert os.path.exists(data_dir)

        concept_paths = pd.read_csv(os.path.join(data_dir, 'image_dicarlo_hvm-public.csv'))
        concept_paths = concept_paths[['object_name', 'filename']]
        concept_paths['filename'] = data_dir + '/' + concept_paths['filename']
        concept_paths = concept_paths.groupby('object_name')['filename'].agg(list)
        concept_paths = concept_paths.values.tolist()
        self.concept_paths = concept_paths

    def get_image_concept_paths(self) -> List[List[str]]:
        return self.concept_paths


def neural_assembly_manifold_statistics(assembly: NeuroidAssembly, concept_coord: str):
    concept_manifolds = [g[1].values for g in assembly.groupby(concept_coord)]
    concept_manifolds = [ManifoldGeometry(activations)
                         for activations in tqdm(concept_manifolds, desc='concept manifolds')]
    progress = tqdm(total=1, desc="manifold statistics")
    manifold_statistics = get_manifold_statistics(concept_manifolds)
    progress.update(1)
    progress.close()
    return manifold_statistics
