import os
import torch
from torch.nn import functional as F
from model_tools.activations.core import change_dict
from model_tools.activations.pytorch import PytorchWrapper


class GlobalMaxPool2d:
    def __init__(self, activations_extractor: PytorchWrapper):
        self._extractor = activations_extractor

    def __call__(self, batch_activations):
        def apply_pool(layer, activations):
            if activations.ndim != 4:
                return activations
            activations = torch.from_numpy(activations)
            activations = F.adaptive_max_pool2d(activations, 1)
            activations = activations.numpy()
            return activations

        return change_dict(batch_activations, apply_pool, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

    @classmethod
    def hook(cls, activations_extractor):
        hook = GlobalMaxPool2d(activations_extractor=activations_extractor)
        assert not cls.is_hooked(activations_extractor), "GlobalMaxPool2d already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())
