import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import brainscore.benchmarks as bench
from brainscore.metrics.regression import linear_regression
from dimensionality import get_results
from activations_models.generators import unsup_vvs_generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unsupervised activations_models on MajajHong2015.')
    parser.add_argument('--debug', action='store_true',
                        help='run only a single model and layer')
    parser.add_argument('--regression', type=str, default='pls',
                        help='regression type for fitting neural data', choices=['pls', 'lin'])
    parser.add_argument('--n_components', type=int, default=1000,
                        help='number of PCA components prior to fitting encoder (-1 for no PCA)')
    args = parser.parse_args()

    if args.n_components == -1:
        args.n_components = None    # no PCA

    regions = ['IT', 'V4'] if not args.debug else ['IT']

    results = pd.DataFrame()
    for region in regions:
        benchmark_identifier = f'dicarlo.MajajHong2015public.{region}-pls'
        benchmark = bench.load(benchmark_identifier)
        if args.regression == 'lin':
            benchmark._identifier = benchmark.identifier.replace('pls', args.regression)
            benchmark._similarity_metric.regression = linear_regression()

        result = get_results(benchmark, unsup_vvs_generator(debug=args.debug), args.n_components)
        result = result.assign(region=region)
        results = results.append(result)

    results.to_csv(f'results_majajhong2015_{args.regression}.csv', index=False)
