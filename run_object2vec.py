import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from bonnerlab_brainscore.benchmarks.object2vec import Object2VecEncoderBenchmark
from dimensionality import get_results
from activations_models.generators import unsup_vvs_generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run unsupervised activations_models on Object2Vec.')
    parser.add_argument('--debug', action='store_true',
                        help='run only a single model and layer')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='path to the Object2Vec dataset directory')
    parser.add_argument('--regression', type=str, default='pls',
                        help='regression type for fitting neural data', choices=['pls', 'lin'])
    parser.add_argument('--n_components', type=int, default=1000,
                        help='number of PCA components prior to fitting encoder (-1 for no PCA)')
    args = parser.parse_args()

    if args.n_components == -1:
        args.n_components = None    # no PCA

    benchmark = Object2VecEncoderBenchmark(data_dir=args.data_dir, regression=args.regression,
                                           regions=['EVC', 'OPA', 'PPA', 'LOC', 'PFS', 'OFA', 'FFA'])

    results = get_results(benchmark, unsup_vvs_generator(debug=args.debug), args.n_components)
    results = results.assign(region='+'.join(benchmark.regions))

    results.to_csv(f'results_object2vec_{args.regression}.csv', index=False)
