import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.decomposition import PCA
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from model_tools.activations.pca import LayerPCA
from model_tools.brain_transformation.neural import LayerScores
from result_caching import store_dict, store_xarray


def get_results(benchmark, models_generator, n_components):
    try:
        stimulus_identifier = benchmark._assembly.stimulus_set_identifier
    except AttributeError:
        stimulus_identifier = benchmark.stimulus_set.identifier

    results = pd.DataFrame()

    for activations_model, layers in models_generator:
        if n_components is not None:
            activations_model.identifier += f'-{n_components}components'

        result = get_model_scores(activations_model, benchmark, layers, n_components)

        effdims_dataset = get_model_effdims_dataset(activations_model, stimulus_identifier, layers)
        result = pd.merge(result, effdims_dataset, on=['model', 'layer'])

        if n_components is not None:
            effdims_imagenet = get_model_effdims_imagenet(activations_model, n_components)
            result = pd.merge(result, effdims_imagenet, on=['model', 'layer'])

        results = results.append(result)

    return results


def get_model_scores(activations_model, benchmark, layers, n_components):
    tf.reset_default_graph()

    if n_components is not None:
        _ = LayerPCA.hook(activations_model, n_components=n_components)

    model_scores = LayerScores(model_identifier=activations_model.identifier,
                               activations_model=activations_model,
                               visual_degrees=8)
    score = model_scores(benchmark=benchmark, layers=layers, prerun=True)

    if 'aggregation' in score.dims:
        score = score.to_dataframe(name='').unstack(level='aggregation').reset_index()
        score.columns = ['layer', 'score', 'score_error']
    else:
        score = score.to_dataframe(name='').reset_index()
        score.columns = ['layer', 'score']
    score = score.assign(model=activations_model.identifier, benchmark=benchmark.identifier)

    return score


def get_model_effdims_imagenet(activations_model, n_components):
    function_identifier = f'{LayerPCA.__module__}.{LayerPCA.__name__}._pcas/' \
                          f'identifier={activations_model.identifier},n_components={n_components}'
    store = store_dict(dict_key='layers', identifier_ignore=['layers'])
    pcas = store.load(function_identifier)
    effdims = {layer: effective_dimensionality(pca) for layer, pca in pcas.items()}

    effdims = [{'layer': layer, 'effective_dimensionality_imagenet': dim} for layer, dim in effdims.items()]
    effdims = pd.DataFrame(effdims)
    effdims = effdims.assign(model=activations_model.identifier)

    return effdims


@store_dict(dict_key='layers', identifier_ignore=['layers'])
def model_dataset_pcas(model_identifier, stimuli_identifier, layers):
    identifier = f'identifier={model_identifier},stimuli_identifier={stimuli_identifier}'
    function_identifier = f'model_tools.activations.core.ActivationsExtractorHelper._from_paths_stored/{identifier}'
    store = store_xarray(identifier_ignore=['model', 'benchmark', 'layers', 'prerun'],
                         combine_fields={'layers': 'layer'})
    regressors = store.load(function_identifier)
    pcas = {layer: PCA().fit(x) for layer, x in regressors.groupby('layer')}
    return pcas


def get_model_effdims_dataset(activations_model, stimuli_identifier, layers):
    pcas = model_dataset_pcas(activations_model.identifier, stimuli_identifier, layers)
    effdims = {layer: effective_dimensionality(pca) for layer, pca in pcas.items()}
    effdims = [{'layer': layer, 'effective_dimensionality_dataset': dim} for layer, dim in effdims.items()]
    effdims = pd.DataFrame(effdims)
    effdims = effdims.assign(model=activations_model.identifier)
    return effdims


def effective_dimensionality(pca):
    eigen_values = pca.singular_values_ ** 2 / (pca.n_components_ - 1)
    effective_dim = eigen_values.sum() ** 2 / (eigen_values ** 2).sum()
    return effective_dim
