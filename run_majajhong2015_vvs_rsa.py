import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from benchmarks.majajhong2015_rsa import DicarloMajajHong2015ITRDM, DicarloMajajHong2015V4RDM
from dimensionality import get_results
from activations_models.generators import unsup_vvs_generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run base models on MajajHong2015.')
    parser.add_argument('--n_components', type=int, default=1000,
                        help='number of PCA components prior to fitting encoder (-1 for no PCA)')
    args = parser.parse_args()

    if args.n_components == -1:
        args.n_components = None    # no PCA

    regions = ['IT', 'V4']

    results = pd.DataFrame()
    for region in regions:
        if region == 'IT':
            benchmark = DicarloMajajHong2015ITRDM()
        else:
            benchmark = DicarloMajajHong2015V4RDM()

        result = get_results(benchmark, unsup_vvs_generator(), args.n_components)
        result = result.assign(region=region)
        results = results.append(result)

    results.to_csv(f'results/majajhong2015_vvs_rsa.csv', index=False)
