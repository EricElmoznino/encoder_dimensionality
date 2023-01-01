import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import brainscore.benchmarks as bench
from brainscore.metrics.regression import linear_regression, ridge_regression
from bonnerlab_brainscore.benchmarks.object2vec import Object2VecEncoderBenchmark
from model_tools.activations.pca import LayerPCA
from model_tools.brain_transformation.neural import LayerScores
from activation_models.generators import get_activation_models
from custom_model_tools.hooks import GlobalMaxPool2d
from utils import timed, id_to_properties


@timed
def main(benchmark, pooling, debug=False):
    save_path = f'results/encoding|benchmark:{benchmark._identifier}|pooling:{pooling}.csv'
    if os.path.exists(save_path):
        print(f'Results already exists: {save_path}')
        return
    
    scores = pd.DataFrame()
    for model, layers in get_activation_models():
        layer_scores = fit_encoder(benchmark, model, layers, pooling)
        scores = scores.append(layer_scores)
        if debug:
            break
    if not debug:
        scores.to_csv(save_path, index=False)


def fit_encoder(benchmark, model, layers, pooling, hooks=None):
    """Fit layers one at a time to save on memory"""

    layer_scores = pd.DataFrame()
    model_identifier = model.identifier
    model_properties = id_to_properties(model_identifier)

    for layer in layers:
        if pooling:
            handle = GlobalMaxPool2d.hook(model)
            model.identifier = model_identifier + f'|layer:{layer}|pooling:True'
        else:
            handle = LayerPCA.hook(model, n_components=1000)
            model.identifier = model_identifier + f'|layer:{layer}|pooling:False|n_components:1000'

        handles = []
        if hooks is not None:
            handles = [cls.hook(model) for cls in hooks]

        model_scores = LayerScores(model_identifier=model.identifier,
                                   activations_model=model,
                                   visual_degrees=8)
        score = model_scores(benchmark=benchmark, layers=[layer], prerun=True)
        handle.remove()

        for h in handles:
            h.remove()

        if 'aggregation' in score.dims:
            score = score.to_dataframe(name='').unstack(level='aggregation').reset_index()
            score.columns = ['layer', 'score', 'score_error']
        else:
            score = score.to_dataframe(name='').reset_index()
            score.columns = ['layer', 'score']

        layer_scores = layer_scores.append(score)

    layer_scores = layer_scores.assign(**model_properties)
    return layer_scores


def get_benchmark(benchmark, region, regression, data_dir):
    if benchmark == 'majajhong2015':
        assert region in ['IT', 'V4']
        identifier = f'dicarlo.MajajHong2015public.{region}-pls'
        benchmark = bench.load(identifier)
        if regression == 'lin':
            benchmark._identifier = benchmark.identifier.replace('pls', 'lin')
            benchmark._similarity_metric.regression = linear_regression()
            benchmark._similarity_metric.regression._regression.alpha = 0.1
        elif regression == 'l2':
            benchmark._identifier = benchmark.identifier.replace('pls', 'l2')
            benchmark._similarity_metric.regression = ridge_regression()
    elif benchmark == 'freeman2013':
        assert region == 'V1'
        identifier = f'movshon.FreemanZiemba2013public.{region}-pls'
        benchmark = bench.load(identifier)
        if regression == 'lin':
            benchmark._identifier = benchmark.identifier.replace('pls', 'lin')
            benchmark._similarity_metric.regression = linear_regression()
            benchmark._similarity_metric.regression._regression.alpha = 0.1
        elif regression == 'l2':
            benchmark._identifier = benchmark.identifier.replace('pls', 'l2')
            benchmark._similarity_metric.regression = ridge_regression()
    elif benchmark == 'object2vec':
        if region == 'all':
            region = None
        regions = region if region is None or ',' not in region else region.split(',')
        benchmark = Object2VecEncoderBenchmark(data_dir=data_dir, regions=regions, regression=regression)
    else:
        raise ValueError(f'Unknown benchmark: {benchmark}')
    return benchmark


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit encoding models to a neural dataset')
    parser.add_argument('--bench', type=str, default='majajhong2015',
                        choices=['majajhong2015', 'freeman2013', 'object2vec'],
                        help='Neural benchmark dataset to fit')
    parser.add_argument('--region', type=str, default='IT',
                        help='Region(s) to fit. Valid region(s) depend on the neural benchmark')
    parser.add_argument('--regression', type=str, default='pls',
                        choices=['pls', 'lin', 'l2'],
                        help='Partial-least-squares or ordinary-least-squares for fitting')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory for neural benchmark (only required for "object2vec")')
    parser.add_argument('--no_pooling', dest='pooling', action='store_false',
                        help='Do not perform global max-pooling prior to fitting')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    benchmark = get_benchmark(benchmark=args.bench, region=args.region,
                              regression=args.regression, data_dir=args.data_dir)
    main(benchmark=benchmark, pooling=args.pooling, debug=args.debug)
