import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools
from sklearn.decomposition import PCA
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import brainscore.benchmarks as bench
from brainscore.metrics.regression import linear_regression
from candidate_models.base_models.unsupervised_vvs import ModelBuilder
from model_tools.activations.pca import LayerPCA
from model_tools.brain_transformation.neural import LayerScores
from result_caching import store_dict, store_xarray

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


def get_model_scores(model_identifier, regression, layers, region, n_components):
    tf.reset_default_graph()

    activations_model = ModelBuilder()(model_identifier)
    activations_model.identifier += '-majajhong2015'
    if n_components is not None:
        activations_model.identifier += f'-{n_components}components'
        _ = LayerPCA.hook(activations_model, n_components=n_components)

    benchmark_identifier = f'dicarlo.MajajHong2015public.{region}-pls'
    benchmark = bench.load(benchmark_identifier)
    if regression == 'lin':
        benchmark._identifier = benchmark.identifier.replace('pls', regression)
        benchmark._similarity_metric.regression = linear_regression()

    model_scores = LayerScores(model_identifier=activations_model.identifier,
                               activations_model=activations_model,
                               visual_degrees=8)
    score = model_scores(benchmark=benchmark, layers=layers)

    score = score.to_dataframe(name='').unstack(level='aggregation').reset_index()
    score.columns = ['layer', 'score', 'score_error']
    score = score.assign(model=model_identifier, region=region)

    return score


def get_model_effdims_imagenet(model_identifier, n_components):
    activations_model_identifier = model_identifier + '-majajhong2015'
    if n_components is not None:
        activations_model_identifier = activations_model_identifier + f'-{n_components}components'
    function_identifier = f'{LayerPCA.__module__}.{LayerPCA.__name__}._pcas/' \
                          f'identifier={activations_model_identifier},n_components={n_components}'
    store = store_dict(dict_key='layers', identifier_ignore=['layers'])
    pcas = store.load(function_identifier)
    effdims = {layer: effective_dimensionality(pca) for layer, pca in pcas.items()}

    effdims = [{'layer': layer, 'effective_dimensionality_imagenet': dim} for layer, dim in effdims.items()]
    effdims = pd.DataFrame(effdims)
    effdims = effdims.assign(model=model_identifier)

    return effdims


def get_model_effdims_dataset(model_identifier, n_components):
    activations_model_identifier = model_identifier + '-majajhong2015'
    if n_components is not None:
        activations_model_identifier = activations_model_identifier + f'-{n_components}components'
    stimuli_identifier = 'dicarlo.hvm-public'
    function_identifier = 'model_tools.activations.core.ActivationsExtractorHelper._from_paths_stored/' \
                          f'identifier={activations_model_identifier},stimuli_identifier={stimuli_identifier}'
    store = store_xarray(identifier_ignore=['model', 'benchmark', 'layers', 'prerun'],
                         combine_fields={'layers': 'layer'})
    regressors = store.load(function_identifier)
    pcas = {layer: PCA().fit(x) for layer, x in regressors.groupby('layer')}
    effdims = {layer: effective_dimensionality(pca) for layer, pca in pcas.items()}

    effdims = [{'layer': layer, 'effective_dimensionality_dataset': dim} for layer, dim in effdims.items()]
    effdims = pd.DataFrame(effdims)
    effdims = effdims.assign(model=model_identifier)

    return effdims


def effective_dimensionality(pca):
    eigen_values = pca.singular_values_ ** 2 / (pca.n_components_ - 1)
    effective_dim = eigen_values.sum() ** 2 / (eigen_values ** 2).sum()
    return effective_dim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unsupervised models on BrainScore.')
    parser.add_argument('--debug', action='store_true',
                        help='run only a single model and layer')
    parser.add_argument('--regression', type=str, default='pls',
                        help='regression type for fitting neural data', choices=['pls', 'lin'])
    parser.add_argument('--n_components', type=int, default=1000,
                        help='number of PCA components prior to fitting encoder (-1 for no PCA)')
    args = parser.parse_args()

    if args.debug:
        vvs_models = ['resnet18-supervised']
        regions = ['IT']
    if args.n_components == -1:
        args.n_components = None

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

        result = get_model_scores(model_identifier, args.regression, layers, region, args.n_components)
        effdims_dataset = get_model_effdims_dataset(model_identifier, args.n_components)
        result = pd.merge(result, effdims_dataset, on=['model', 'layer'])
        if args.n_components is not None:
            effdims_imagenet = get_model_effdims_imagenet(model_identifier, args.n_components)
            result = pd.merge(result, effdims_imagenet, on=['model', 'layer'])

        results = results.append(result)

    results.to_csv(f'results_{args.regression}.csv', index=False)
