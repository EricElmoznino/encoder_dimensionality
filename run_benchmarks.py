import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import brainscore.benchmarks as bench
from candidate_models.base_models.unsupervised_vvs import ModelBuilder
from model_tools.activations.pca import LayerPCA
from model_tools.brain_transformation.neural import LayerScores
from result_caching import store_dict

tf_res18_layers = ['encode_1.conv'] + ['encode_%i' % i for i in range(1, 10)]
pt_resnet18_layers = ['relu', 'maxpool'] +\
                     ['layer1.0.relu', 'layer1.1.relu'] +\
                     ['layer2.0.relu', 'layer2.1.relu'] +\
                     ['layer3.0.relu', 'layer3.1.relu'] +\
                     ['layer4.0.relu', 'layer4.1.relu']
prednet_layers = ['A_%i' % i for i in range(1, 4)] \
                 + ['Ahat_%i' % i for i in range(1, 4)] \
                 + ['E_%i' % i for i in range(1, 4)] \
                 + ['R_%i' % i for i in range(1, 4)]
vvs_models = ['resnet18-supervised', 'resnet18-la', 'resnet18-ir', 'resnet18-ae',
              'resnet18-cpc', 'resnet18-color', 'resnet18-rp', 'resnet18-depth',
              'resnet18-simclr', 'resnet18-deepcluster', 'resnet18-cmc']
regions = ['IT', 'V4']


def get_model_scores(model_identifier, layers, region, n_components):
    tf.reset_default_graph()
    activations_model = ModelBuilder()(model_identifier)
    activations_model.identifier += '-majajhong2015'
    benchmark_identifier = f'dicarlo.MajajHong2015public.{region}-pls'
    _ = LayerPCA.hook(activations_model, n_components=n_components)

    model_scores = LayerScores(model_identifier=activations_model.identifier,
                               activations_model=activations_model,
                               visual_degrees=8)
    score = model_scores(benchmark=bench.load(benchmark_identifier),
                         layers=layers)

    score = score.to_dataframe(name='').unstack(level='aggregation').reset_index()
    score.columns = ['layer', 'score', 'score_error']
    score = score.assign(model=model_identifier, region=region)

    return score


def get_model_effdims(model_identifier, n_components):
    activations_model_identifier = model_identifier + '-majajhong2015'
    store = store_dict(dict_key='layers', identifier_ignore=['layers'])
    function_identifier = f'{LayerPCA.__module__}.{LayerPCA.__name__}._pcas/' \
                          f'identifier={activations_model_identifier},n_components={n_components}'
    pcas = store.load(function_identifier)
    effdims = {layer: effective_dimensionality(pca) for layer, pca in pcas.items()}

    effdims = [{'layer': layer, 'effective_dimensionality': dim} for layer, dim in effdims.items()]
    effdims = pd.DataFrame(effdims)
    effdims = effdims.assign(model=model_identifier, region=region)

    return effdims


def effective_dimensionality(pca):
    eigen_values = pca.singular_values_ ** 2 / (pca.n_components_ - 1)
    effective_dim = eigen_values.sum() ** 2 / (eigen_values ** 2).sum()
    return effective_dim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unsupervised models on BrainScore.')
    parser.add_argument('--debug', action='store_true',
                        help='run only a single model and layer')
    parser.add_argument('--n_components', type=int, default=1000,
                        help='number of PCA components prior to fitting encoder')
    args = parser.parse_args()

    if args.debug:
        vvs_models = ['resnet18-supervised']
        regions = ['IT']

    results = pd.DataFrame()
    for model_identifier, region in itertools.product(vvs_models, regions):
        if model_identifier in ModelBuilder.PT_MODELS:
            layers = pt_resnet18_layers
        elif model_identifier == 'prednet':
            layers = prednet_layers
        elif model_identifier == 'resnet18-simclr':
            layers = tf_res18_layers[1:]
        else:
            layers = tf_res18_layers
        if args.debug:
            layers = layers[:1]

        model_scores = get_model_scores(model_identifier, layers, region, args.n_components)
        model_effdims = get_model_effdims(model_identifier, args.n_components)
        result = pd.merge(model_scores, model_effdims, on=['model', 'layer', 'region'])

        results = results.append(result)

    results.to_csv('results.csv', index=False)
