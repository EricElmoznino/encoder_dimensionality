# Figure 5 of https://www.biorxiv.org/content/10.1101/2021.03.21.436284v1.full.pdf
# Original code found in https://github.com/bsorsch/geometry_fewshot_learning/blob/master/ResNet_macaque_stimuli.ipynb
#
# Parameters:
#     - Dataset: majaj_2015 (brainscore IT object dataset)
#     - Number of objects: 64
#     - Number of images per object: 50

import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from reproduce_literature.generators import get_gcl_fig5_models
from custom_model_tools.manifold import LayerManifoldStatisticsMajajHong2015
from utils import timed


@timed
def main(pooling, debug=False):
    manifold_statistics_df = pd.DataFrame()
    for model, layers in get_gcl_fig5_models():
        manifold_statistics = LayerManifoldStatisticsMajajHong2015(activations_extractor=model,
                                                                   pooling=pooling)
        manifold_statistics.fit(layers)
        manifold_statistics_df = manifold_statistics_df.append(manifold_statistics.as_df())
        if debug:
            break
    if not debug:
        manifold_statistics_df.to_csv(f'reproduce_literature/results/gcl_manifolds|pooling:{pooling}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce manifold statistics data for GCL Figure 5')
    parser.add_argument('--no_pooling', dest='pooling', action='store_false',
                        help='Do not perform global max-pooling prior to computing the manifold statistics')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    main(pooling=args.pooling, debug=args.debug)
