import os
import torch
from torch.nn import functional as F
from model_tools.activations.core import change_dict
from model_tools.activations.pytorch import PytorchWrapper
from typing import Dict


class ZScore:
    def __init__(self, activations_extractor: PytorchWrapper, dim=1):
        assert activations_extractor._model.zscore
        self._extractor = activations_extractor
        self._dim = dim

    def __call__(self, batch_activations):
        def apply_zscore(layer, activations):
            std = activations.std(self._dim, keepdims=True)
            std[std < 1e-9] = 1
            activations = (activations - activations.mean(self._dim, keepdims=True)) / std
            return activations

        return change_dict(batch_activations, apply_zscore, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

    @classmethod
    def hook(cls, activations_extractor, dim=1):
        hook = ZScore(activations_extractor=activations_extractor, dim=dim)
        assert not cls.is_hooked(activations_extractor), "ZScore already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())


class MaxPool2d:
    def __init__(self, activations_extractor: PytorchWrapper, pool_map: Dict[str, int]):
        assert activations_extractor._model.zscore
        self._extractor = activations_extractor
        self._pool_map = pool_map

    def __call__(self, batch_activations):
        def apply_pool(layer, activations):
            if layer not in self._pool_map:
                return activations
            pool_size = self._pool_map[layer]
            activations = torch.from_numpy(activations)
            activations = F.max_pool2d(activations, pool_size)
            activations = activations.numpy()
            return activations

        return change_dict(batch_activations, apply_pool, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

    @classmethod
    def hook(cls, activations_extractor, pool_map: Dict[str, int]):
        hook = MaxPool2d(activations_extractor=activations_extractor, pool_map=pool_map)
        assert not cls.is_hooked(activations_extractor), "MaxPool2d already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())
