from __future__ import annotations
import logging
import os.path
from result_caching import store_dict
from sklearn.metrics import top_k_accuracy_score, label_ranking_average_precision_score, log_loss
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.special import softmax
import pandas as pd
import numpy as np
from tqdm import tqdm
from model_tools.activations.core import flatten
from model_tools.utils import fullname
from custom_model_tools.hooks import GlobalMaxPool2d, GlobalAvgPool2d, RandomProjection
from utils import id_to_properties, get_imagenet_val
from typing import List, Dict


class NShotLearningBase:

    def __init__(self, activations_extractor, classifier,
                 n_cats=50, n_train=(1, 5, 20, 50), n_test=50,
                 n_repeats=10, pooling='max', stimuli_identifier=None):
        assert classifier in ['linear', 'prototype', 'maxmargin']
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._classifier = classifier
        self._n_cats = n_cats
        self._n_train = n_train
        self._n_test = n_test
        self._n_repeats = n_repeats
        self._pooling = pooling
        self._stimuli_identifier = stimuli_identifier
        self._layer_performance_statistics = {}

    def fit(self, layers):
        self._layer_performance_statistics = self._fit(identifier=self._extractor.identifier,
                                                       stimuli_identifier=self._stimuli_identifier,
                                                       classifier=self._classifier,
                                                       layers=layers,
                                                       pooling=self._pooling)

    def as_df(self):
        df = pd.DataFrame()
        for layer, statistics in self._layer_performance_statistics.items():
            for statistic in statistics:
                statistic['layer'] = layer
            df = df.append(statistics, ignore_index=True)
        properties = id_to_properties(self._extractor.identifier)
        df = df.assign(**properties)
        return df

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def _fit(self, identifier, stimuli_identifier, classifier, layers, pooling):
        n_samples = max(self._n_train) + self._n_test
        cat_paths = self.get_image_paths(self._n_cats, n_samples)

        # Compute classification statistics for every layer individually to save on memory.
        # This is more inefficient because we run images through the network several times,
        # but it is a more scalable approach when using many images and large layers.
        layer_performance_statistics = {}
        for layer in layers:
            if pooling == "max":
                handle = GlobalMaxPool2d.hook(self._extractor)
            elif pooling == "avg":
                handle = GlobalAvgPool2d.hook(self._extractor)
            elif pooling == "none":
                handle = RandomProjection.hook(self._extractor)
            else:
                raise ValueError(f"Unknown pooling method {pooling}")

            self._logger.debug('Retrieving activations')
            activations = self._extractor([path for cat in cat_paths for path in cat], layers=[layer])
            activations = activations.sel(layer=layer).values
            activations = flatten(activations)
            activations = activations.reshape(self._n_cats, n_samples, -1)

            self._logger.debug('Training/evaluating classifiers')
            performance_statistics = []

            sample_orders = np.arange(n_samples)
            for i_repeat in tqdm(range(self._n_repeats), desc='repeat'):
                np.random.seed(i_repeat)
                np.random.shuffle(sample_orders)

                X_test = activations[:, sample_orders[-self._n_test:], :]
                y_test = np.ones((self._n_cats, self._n_test), dtype=int) * \
                         np.arange(self._n_cats).reshape(-1, 1)
                X_test = X_test.reshape(-1, X_test.shape[-1])
                y_test = y_test.reshape(-1)
                for n_train in self._n_train:
                    X_train = activations[:, sample_orders[:n_train], :]
                    y_train = np.ones((self._n_cats, n_train), dtype=int) * \
                              np.arange(self._n_cats).reshape(-1, 1)
                    X_train = X_train.reshape(-1, X_train.shape[-1])
                    y_train = y_train.reshape(-1)

                    performance = self.classifier_performance(X_train, y_train, X_test, y_test)
                    performance['n_train'] = n_train
                    performance['i_repeat'] = i_repeat
                    performance_statistics.append(performance)

            layer_performance_statistics[layer] = performance_statistics

            handle.remove()

        return layer_performance_statistics

    def classifier_performance(self, X_train, y_train, X_test, y_test) -> Dict[str, float]:
        if self._classifier == 'linear':
            return logistic_performance(X_train, y_train, X_test, y_test)
        elif self._classifier == 'prototype':
            return prototype_performance(X_train, y_train, X_test, y_test)
        elif self._classifier == 'maxmargin':
            return maxmargin_performance(X_train, y_train, X_test, y_test)
        else:
            raise ValueError(f'Unknown classifier {self._classifier}')

    def get_image_paths(self, n_cats, n_samples) -> List[List[str]]:
        raise NotImplementedError()


class NShotLearningImageNet(NShotLearningBase):

    def __init__(self, *args, **kwargs):
        super(NShotLearningImageNet, self).__init__(*args, **kwargs, stimuli_identifier='imagenet')

    def get_image_paths(self, n_cats, n_samples) -> List[List[str]]:
        return get_imagenet_val(num_classes=n_cats, num_per_class=n_samples, separate_classes=True)


class NShotLearningImageFolder(NShotLearningBase):

    def __init__(self, data_dir, *args, **kwargs):
        super(NShotLearningImageFolder, self).__init__(*args, **kwargs)

        assert os.path.exists(data_dir)
        self.data_dir = data_dir

        cat_paths = []
        cats = os.listdir(data_dir)
        assert len(cats) >= self._n_cats
        cats = cats[:self._n_cats]
        n_samples = max(self._n_train) + self._n_test
        for cat in cats:
            cat_dir = os.path.join(data_dir, cat)
            files = os.listdir(cat_dir)
            assert len(files) >= n_samples
            files = files[:n_samples]
            paths = [os.path.join(cat_dir, file) for file in files]
            cat_paths.append(paths)
        self.cat_paths = cat_paths

    def get_image_paths(self, n_cats, n_samples) -> List[List[str]]:
        return self.cat_paths


class NShotLearningObject2Vec(NShotLearningImageFolder):

    def __init__(self, data_dir, *args, **kwargs):
        data_dir = os.path.join(data_dir, 'stimuli_rgb')
        super(NShotLearningObject2Vec, self).__init__(data_dir, *args, **kwargs,
                                                      n_train=(1, 5, 20, 50), n_test=31,
                                                      stimuli_identifier='object2vec')


def logistic_performance(X_train, y_train, X_test, y_test) -> Dict[str, float]:
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)

        top1 = top_k_accuracy_score(y_test, y_pred, k=1)
        top5 = top_k_accuracy_score(y_test, y_pred, k=5)
        mrr = label_ranking_average_precision_score(label_binarize(y_test, classes=range(y_pred.shape[1])), y_pred)
        ll = -log_loss(y_test, y_pred)

        return {'accuracy (top 1)': top1,
                'accuracy (top 5)': top5,
                'MRR': mrr,
                'log likelihood': ll}


def prototype_performance(X_train, y_train, X_test, y_test) -> Dict[str, float]:
        model = NearestCentroidDistances()
        model.fit(X_train, y_train)
        y_pred = model.predict_distances(X_test)
        y_pred = softmax(-y_pred, axis=1)   # Simply to order classes based on distance (i.e. not real probabilities)

        top1 = top_k_accuracy_score(y_test, y_pred, k=1)
        top5 = top_k_accuracy_score(y_test, y_pred, k=5)
        mrr = label_ranking_average_precision_score(label_binarize(y_test, classes=range(y_pred.shape[1])), y_pred)

        return {'accuracy (top 1)': top1,
                'accuracy (top 5)': top5,
                'MRR': mrr}


def maxmargin_performance(X_train, y_train, X_test, y_test) -> Dict[str, float]:
        model = LinearSVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        top1 = top_k_accuracy_score(y_test, y_pred, k=1)

        return {'accuracy (top 1)': top1}


class NearestCentroidDistances(NearestCentroid):
    def predict_distances(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr')
        distances = pairwise_distances(X, self.centroids_, metric=self.metric)
        return distances
