import os
import numpy as np
import torch
from torch.nn import functional as F
from model_tools.activations.core import change_dict


class GlobalMaxPool2d:
    def __init__(self, activations_extractor):
        self._extractor = activations_extractor

    def __call__(self, batch_activations):
        def apply(layer, activations):
            if activations.ndim != 4:
                return activations
            activations = torch.from_numpy(activations)
            activations = F.adaptive_max_pool2d(activations, 1)
            activations = activations.numpy()
            return activations

        return change_dict(batch_activations, apply, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

    @classmethod
    def hook(cls, activations_extractor):
        hook = GlobalMaxPool2d(activations_extractor)
        assert not cls.is_hooked(activations_extractor), "GlobalMaxPool2d already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())


class RandomProjection:
    def __init__(self, activations_extractor, n_components=1024):
        self._extractor = activations_extractor
        self.n_components = n_components
        self.layer_ws = {}
        np.random.seed(27)

    def __call__(self, batch_activations):
        def apply(layer, activations):
            activations = activations.reshape(activations.shape[0], -1)
            if layer not in self.layer_ws:
                w = np.random.rand(activations.shape[-1], self.n_components) / np.sqrt(self.n_components)
                self.layer_ws[layer] = w
            else:
                w = self.layer_ws[layer]
            activations = activations @ w
            return activations

        return change_dict(batch_activations, apply, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

    @classmethod
    def hook(cls, activations_extractor):
        hook = RandomProjection(activations_extractor=activations_extractor)
        assert not cls.is_hooked(activations_extractor), "RandomProjection already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())
