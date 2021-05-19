import functools
from torch import nn
from torchvision import transforms
import numpy as np
from model_tools.activations.pytorch import PytorchWrapper, load_preprocess_images, load_images
from .hooks import ZScore, MaxPool2d
from typing import Callable, List, Tuple, Dict


class Model(nn.Module):

    def __init__(self, zscore: bool = False, pool_map: Dict[str, int] = None):
        super(Model, self).__init__()
        self.zscore = zscore
        self.pool_map = pool_map

    def make_wrapper(self) -> Tuple[PytorchWrapper, List[str]]:
        activations_model = PytorchWrapper(model=self,
                                           preprocessing=self.preprocess_func(),
                                           identifier=self.identifier())
        if self.zscore:
            _ = ZScore.hook(activations_model)
        if self.pool_map is not None:
            _ = MaxPool2d.hook(activations_model, self.pool_map)
        return activations_model, self.layers()

    def preprocess_func(self) -> Callable[[List[str]], np.ndarray]:
        if self.model_type() == 'supervised' or self.model_type() == 'unsupervised':
            preprocessing = functools.partial(load_preprocess_images, image_size=224)
        elif self.model_type() == 'engineered':
            def preprocessing(image_filepaths):
                images = load_images(image_filepaths)
                transform = transforms.Compose([
                    transforms.Resize((96, 96)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5)
                ])
                images = [transform(image) for image in images]
                images = np.stack(images)
                return images
        else:
            raise NotImplementedError()

        return preprocessing

    def identifier(self) -> str:
        name = [
            f'base={self.base_name()}',
            f'type={self.model_type()}'
        ]
        if self.zscore:
            name.append('zscore')
        return '-'.join(name)

    def base_name(self) -> str:
        raise NotImplementedError()

    def model_type(self) -> str:
        raise NotImplementedError()

    def layers(self) -> List[str]:
        raise NotImplementedError()
