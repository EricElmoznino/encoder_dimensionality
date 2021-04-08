import os
from model_tools.activations.core import change_dict
from model_tools.activations.pytorch import PytorchWrapper


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
