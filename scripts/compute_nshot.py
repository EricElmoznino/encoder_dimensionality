import argparse
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from activation_models.generators import get_activation_models
from custom_model_tools.n_shot_learning import NShotLearningImageFolder, NShotLearningObject2Vec, NShotLearningImageNet
from utils import timed


@timed
def main(dataset, data_dir, classifier, pooling, debug=False):
    n_shot_df = pd.DataFrame()
    for model, layers in get_activation_models():
        n_shot = get_n_shot_performance(dataset, data_dir, classifier, model, pooling)
        n_shot.fit(layers)
        n_shot_df = n_shot_df.append(n_shot.as_df())
        if debug:
            break
    if not debug:
        n_shot_df.to_csv(f'results/n-shot|dataset:{dataset}|classifier:{classifier}|pooling:{pooling}.csv', index=False)


def get_n_shot_performance(dataset, data_dir, classifier, activations_extractor, pooling):
    if dataset == 'imagenet':
        return NShotLearningImageNet(activations_extractor=activations_extractor,
                                     classifier=classifier,
                                     pooling=pooling)
    elif dataset == 'imagenet21k':
        return NShotLearningImageFolder(data_dir=data_dir,
                                        activations_extractor=activations_extractor,
                                        classifier=classifier,
                                        pooling=pooling,
                                        stimuli_identifier='imagenet21k')
    elif dataset == 'object2vec':
        return NShotLearningObject2Vec(data_dir=data_dir,
                                       activations_extractor=activations_extractor,
                                       classifier=classifier,
                                       pooling=pooling)
    else:
        raise ValueError(f'Unknown n-shot learning dataset: {dataset}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and store n-shot learning performance')
    parser.add_argument('--dataset', type=str,
                        choices=['imagenet', 'imagenet21k', 'object2vec'],
                        help='Classification dataset for which to compute n-shot learning performance')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory containing stimuli')
    parser.add_argument('--classifier', type=str, default=None,
                        choices=['linear', 'prototype', 'maxmargin'],
                        help='Type of classifier to use')
    parser.add_argument('--no_pooling', dest='pooling', action='store_false',
                        help='Do not perform global max-pooling prior to computing the n-shot learning performance')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    main(dataset=args.dataset, data_dir=args.data_dir, classifier=args.classifier,
         pooling=args.pooling, debug=args.debug)
