import logging
import os
from result_caching import store_dict
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from tqdm import tqdm
from model_tools.activations.core import flatten
from model_tools.utils import fullname
from custom_model_tools.hooks import GlobalMaxPool2d, GlobalAvgPool2d, RandomProjection
from custom_model_tools.image_transform import ImageDatasetTransformer
from utils import id_to_properties, get_imagenet_val
from typing import Optional, List


class EigenspectrumBase:
    def __init__(
        self,
        activations_extractor,
        pooling="max",
        stimuli_identifier=None,
        image_transform: Optional[ImageDatasetTransformer] = None,
        hooks: Optional[List] = None,
    ):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._pooling = pooling
        self._hooks = hooks
        self._stimuli_identifier = stimuli_identifier
        self._image_transform = image_transform
        self._layer_eigenspectra = {}

    def fit(self, layers):
        transform_name = (
            None if self._image_transform is None else self._image_transform.name
        )
        self._layer_eigenspectra = self._fit(
            identifier=self._extractor.identifier,
            stimuli_identifier=self._stimuli_identifier,
            layers=layers,
            pooling=self._pooling,
            image_transform_name=transform_name,
        )

    def effective_dimensionalities(self):
        effdims = {
            layer: eigspec.sum() ** 2 / (eigspec**2).sum()
            for layer, eigspec in self._layer_eigenspectra.items()
        }
        return effdims

    def eighty_percent_var(self):
        eighty_percent_var = {}
        for layer, eigspec in self._layer_eigenspectra.items():
            pvar = eigspec.cumsum() / eigspec.sum()
            for i in range(len(pvar)):
                if pvar[i] >= 0.8:
                    eighty_percent_var[layer] = i + 1
                    break
        return eighty_percent_var

    def powerlaw_exponent(self):
        alpha = {}
        for layer, eigspec in self._layer_eigenspectra.items():
            start = 0
            end = np.log10(len(eigspec))
            eignum = np.logspace(start, end, num=50).round().astype(int)
            eigspec = eigspec[eignum - 1]
            logeignum = np.log10(eignum)
            logeigspec = np.log10(eigspec)
            linear_fit = LinearRegression().fit(logeignum.reshape(-1, 1), logeigspec)
            alpha[layer] = -linear_fit.coef_.item()
        return alpha

    def as_df(self):
        df = pd.DataFrame()
        for layer, eigspec in self._layer_eigenspectra.items():
            layer_df = pd.DataFrame(
                {"n": range(1, len(eigspec) + 1), "variance": eigspec}
            )
            layer_df = layer_df.assign(layer=layer)
            df = df.append(layer_df)
        properties = id_to_properties(self._extractor.identifier)
        df = df.assign(**properties)
        return df

    def metrics_as_df(self):
        effdims = self.effective_dimensionalities()
        eightyvar = self.eighty_percent_var()
        alpha = self.powerlaw_exponent()
        df = pd.DataFrame()
        for layer in self._layer_eigenspectra:
            df = df.append(
                {
                    "layer": layer,
                    "effective dimensionality": effdims[layer],
                    "80% variance": eightyvar[layer],
                    "alpha": alpha[layer],
                },
                ignore_index=True,
            )
        properties = id_to_properties(self._extractor.identifier)
        df = df.assign(**properties)
        return df

    @store_dict(dict_key="layers", identifier_ignore=["layers"])
    def _fit(
        self, identifier, stimuli_identifier, layers, pooling, image_transform_name
    ):
        image_paths = self.get_image_paths()
        if self._image_transform is not None:
            image_paths = self._image_transform.transform_dataset(
                self._stimuli_identifier, image_paths
            )

        # Compute activations and PCA for every layer individually to save on memory.
        # This is more inefficient because we run images through the network several times,
        # but it is a more scalable approach when using many images and large layers.
        layer_eigenspectra = {}
        for layer in layers:
            if pooling == "max":
                handle = GlobalMaxPool2d.hook(self._extractor)
            elif pooling == "avg":
                handle = GlobalAvgPool2d.hook(self._extractor)
            elif pooling == "none":
                handle = RandomProjection.hook(self._extractor)
            else:
                raise ValueError(f"Unknown pooling method {pooling}")

            handles = []
            if self._hooks is not None:
                handles = [cls.hook(self._extractor) for cls in self._hooks]

            self._logger.debug("Retrieving stimulus activations")
            activations = self._extractor(image_paths, layers=[layer])
            activations = activations.sel(layer=layer).values

            self._logger.debug("Computing principal components")
            progress = tqdm(total=1, desc="layer principal components")
            activations = flatten(activations)
            pca = PCA(random_state=0)
            pca.fit(activations)
            eigenspectrum = pca.explained_variance_
            progress.update(1)
            progress.close()

            layer_eigenspectra[layer] = eigenspectrum

            handle.remove()

            for h in handles:
                h.remove()

        return layer_eigenspectra

    def get_image_paths(self) -> List[str]:
        raise NotImplementedError()


class EigenspectrumImageNet(EigenspectrumBase):
    def __init__(self, *args, num_classes=1000, num_per_class=10, **kwargs):
        super(EigenspectrumImageNet, self).__init__(
            *args, **kwargs, stimuli_identifier="imagenet"
        )
        assert 1 <= num_classes <= 1000 and 1 <= num_per_class <= 100
        self.num_classes = num_classes
        self.num_per_class = num_per_class
        self.image_paths = get_imagenet_val(
            num_classes=num_classes, num_per_class=num_per_class
        )

    def get_image_paths(self) -> List[str]:
        return self.image_paths


class EigenspectrumImageFolder(EigenspectrumBase):
    def __init__(self, data_dir, *args, num_images=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_images = num_images

        assert os.path.exists(data_dir)
        self.data_dir = data_dir

        image_paths = os.listdir(data_dir)
        if num_images is not None:
            assert len(image_paths) >= num_images
            image_paths = image_paths[:num_images]
        image_paths = [os.path.join(data_dir, file) for file in image_paths]
        self.image_paths = image_paths

    def get_image_paths(self) -> List[str]:
        return self.image_paths


class EigenspectrumNestedImageFolder(EigenspectrumBase):
    def __init__(
        self, data_dir, *args, num_folders=None, num_per_folder=None, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.num_folders = num_folders
        self.num_per_folder = num_per_folder

        assert os.path.exists(data_dir)
        self.data_dir = data_dir

        image_paths = []
        cats = os.listdir(self.data_dir)
        if num_folders is not None:
            assert len(cats) >= num_folders
            cats = cats[:num_folders]
        for cat in cats:
            cat_dir = os.path.join(data_dir, cat)
            files = os.listdir(cat_dir)
            if num_per_folder is not None:
                assert len(files) >= num_per_folder
                files = files[:num_per_folder]
            paths = [os.path.join(cat_dir, file) for file in files]
            image_paths += paths
        self.image_paths = image_paths

    def get_image_paths(self) -> List[str]:
        return self.image_paths


class EigenspectrumImageNet21k(EigenspectrumNestedImageFolder):
    def __init__(self, data_dir, *args, num_classes=996, num_per_class=10, **kwargs):
        super().__init__(
            data_dir,
            *args,
            num_folders=num_classes,
            num_per_folder=num_per_class,
            **kwargs,
            stimuli_identifier="imagenet21k",
        )


class EigenspectrumObject2Vec(EigenspectrumNestedImageFolder):
    def __init__(self, data_dir, *args, **kwargs):
        data_dir = os.path.join(data_dir, "stimuli_rgb")
        super().__init__(data_dir, *args, **kwargs, stimuli_identifier="object2vec")


class EigenspectrumMajajHong2015(EigenspectrumImageFolder):
    def __init__(self, *args, **kwargs):
        data_dir = os.getenv("BRAINIO_HOME", os.path.expanduser("~/.brainio"))
        data_dir = os.path.join(data_dir, "image_dicarlo_hvm-public")
        assert os.path.exists(data_dir)
        super().__init__(
            data_dir, *args, **kwargs, stimuli_identifier="dicarlo.hvm-public"
        )

        self.image_paths = [p for p in self.image_paths if p[-4:] == ".png"]
