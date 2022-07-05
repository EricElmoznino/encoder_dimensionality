from abc import ABC, abstractmethod
import logging
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from result_caching import store_dict
import brainscore.benchmarks as bench
from brainscore.benchmarks._neural_common import explained_variance
from model_tools.activations.pca import LayerPCA, _get_imagenet_val
from model_tools.activations.core import flatten, change_dict
from model_tools.utils import fullname
from custom_model_tools.hooks import GlobalMaxPool2d, RandomProjection
from utils import id_to_properties


class DimUsageBase(ABC):

    def __init__(self,
                 activations_extractor,
                 pooling=True,
                 interval=5,
                 max=32):
        assert not LayerPCA.is_hooked(activations_extractor)

        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._pooling = pooling
        self._interval = interval
        self._max = max
        self._layer_results = None

    def fit(self, layers):
        self._layer_results = self._fit(model_identifier=self._extractor.identifier,
                                        benchmark_identifier=self.benchmark_identifier,
                                        layers=layers,
                                        pooling=self._pooling)

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def _fit(self, model_identifier, benchmark_identifier, layers, pooling):
        self._extractor.identifier += '|pcs'
        layer_results = {}
        for layer in layers:
            if pooling:
                handle_pool = GlobalMaxPool2d.hook(self._extractor)
            else:
                handle_pool = RandomProjection.hook(self._extractor)
            handle_pca = LayerPCAForce.hook(self._extractor)

            self._logger.debug('Retrieving stimulus activations')
            activations = self._extractor(self.stimuli, layers=[layer])
            activations = activations.sel(layer=layer)

            self._logger.debug('Fitting with multiple numbers of PCs')
            progress = tqdm(total=min(self._max, activations.sizes['neuroid']) // self._interval + 1,
                            desc="different PCs")
            pc_is, scores = [], []
            for pc_i in range(1, min(self._max, activations.sizes['neuroid']), self._interval):
                pc_activations = activations.isel(neuroid=slice(pc_i))
                score = self.get_score(pc_activations)
                pc_is.append(pc_i)
                scores.append(score)
                progress.update(1)
            progress.close()

            layer_results[layer] = {
                'num_pcs': pc_is,
                'scores': scores
            }

            handle_pca.remove()
            handle_pool.remove()
        self._extractor.identifier = self._extractor.identifier.replace('|pcs', '')

        return layer_results

    def as_df(self):
        assert self._layer_results is not None
        df = pd.DataFrame()
        for layer, results in self._layer_results.items():
            layer_df = pd.DataFrame(results)
            layer_df = layer_df.assign(layer=layer)
            df = df.append(layer_df)
        properties = id_to_properties(self._extractor.identifier)
        df = df.assign(**properties)
        return df

    @property
    @abstractmethod
    def benchmark_identifier(self):
        pass

    @property
    @abstractmethod
    def stimuli(self):
        pass

    @abstractmethod
    def get_score(self, assembly):
        pass


class DimUsageMajajHongIT(DimUsageBase):

    def __init__(self, *args, **kwargs):
        super(DimUsageMajajHongIT, self).__init__(*args, **kwargs)

        self._benchmark = bench.load('dicarlo.MajajHong2015public.IT-pls')

    @property
    def benchmark_identifier(self):
        return self._benchmark._identifier

    @property
    def stimuli(self):
        return self._benchmark._assembly.stimulus_set

    def get_score(self, assembly):
        self._benchmark._similarity_metric.regression._regression.n_components = min(25, assembly.sizes['neuroid'])
        raw_score = self._benchmark._similarity_metric(assembly, self._benchmark._assembly)
        score = explained_variance(raw_score, self._benchmark.ceiling)
        score = score.sel(aggregation='center').item()
        return score

# Helper

class LayerPCAForce:

    def __init__(self, activations_extractor):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._layer_pcas = {}

    def __call__(self, batch_activations):
        self._ensure_initialized(batch_activations.keys())

        def apply_pca(layer, activations):
            pca = self._layer_pcas[layer]
            activations = flatten(activations)
            if pca is None:
                return activations
            return pca.transform(activations)

        return change_dict(batch_activations, apply_pca, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

    def _ensure_initialized(self, layers):
        missing_layers = [layer for layer in layers if layer not in self._layer_pcas]
        if len(missing_layers) == 0:
            return
        layer_pcas = self._pcas(identifier=self._extractor.identifier, layers=missing_layers)
        self._layer_pcas = {**self._layer_pcas, **layer_pcas}

    def _pcas(self, identifier, layers):
        self._logger.debug('Retrieving ImageNet activations')
        imagenet_paths = _get_imagenet_val(num_images=1000)
        self.handle.disable()
        imagenet_activations = self._extractor(imagenet_paths, layers=layers)
        imagenet_activations = {layer: imagenet_activations.sel(layer=layer).values
                                for layer in np.unique(imagenet_activations['layer'])}
        assert len(set(activations.shape[0] for activations in imagenet_activations.values())) == 1, "stimuli differ"
        self.handle.enable()

        self._logger.debug('Computing ImageNet principal components')
        progress = tqdm(total=len(imagenet_activations), desc="layer principal components")

        def init_and_progress(layer, activations):
            activations = flatten(activations)
            pca = PCA(random_state=0)
            pca.fit(activations)
            progress.update(1)
            return pca

        from model_tools.activations.core import change_dict
        layer_pcas = change_dict(imagenet_activations, init_and_progress, keep_name=True,
                                 multithread=os.getenv('MT_MULTITHREAD', '1') == '1')
        progress.close()
        return layer_pcas

    @classmethod
    def hook(cls, activations_extractor):
        hook = LayerPCAForce(activations_extractor=activations_extractor)
        assert not cls.is_hooked(activations_extractor), "PCA already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())
