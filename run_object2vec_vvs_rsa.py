import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from bonnerlab_brainscore.benchmarks.object2vec import Object2VecRSABenchmark
from dimensionality import get_results
from activations_models.generators import unsup_vvs_generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run base models on Object2Vec.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='path to the Object2Vec dataset directory')
    parser.add_argument('--n_components', type=int, default=1000,
                        help='number of PCA components prior to fitting encoder (-1 for no PCA)')
    args = parser.parse_args()

    if args.n_components == -1:
        args.n_components = None    # no PCA

    benchmark = Object2VecRSABenchmark(data_dir=args.data_dir)

    results = get_results(benchmark, unsup_vvs_generator(), args.n_components)
    results = results.assign(region='+'.join(benchmark.regions))

    results.to_csv(f'results/object2vec_vvs_rsa.csv', index=False)
