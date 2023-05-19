import os
import argparse
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from torchvision.transforms import Grayscale
from activation_models.generators import get_activation_models
from custom_model_tools.eigenspectrum import (
    EigenspectrumImageNet,
    EigenspectrumImageNet21k,
    EigenspectrumObject2Vec,
    EigenspectrumMajajHong2015,
)
from custom_model_tools.image_transform import ImageDatasetTransformer
from utils import timed


@timed
def main(dataset, data_dir, pooling, grayscale, debug=False):
    save_paths = {
        "eigspectra": f"results/eigspectra|dataset:{dataset}|pooling:{pooling}|grayscale:{grayscale}.csv",
        "eigmetrics": f"results/eigmetrics|dataset:{dataset}|pooling:{pooling}|grayscale:{grayscale}.csv",
    }
    if os.path.exists(save_paths["eigspectra"]) or os.path.exists(
        save_paths["eigmetrics"]
    ):
        print(
            f'Results already exists: {save_paths["eigspectra"]}\n{save_paths["eigmetrics"]}'
        )
        return

    image_transform = (
        ImageDatasetTransformer("grayscale", Grayscale()) if grayscale else None
    )
    eigspec_df = pd.DataFrame()
    eigmetrics_df = pd.DataFrame()
    for model, layers in get_activation_models():
        eigspec = get_eigenspectrum(dataset, data_dir, model, pooling, image_transform)
        eigspec.fit(layers)
        eigspec_df = eigspec_df.append(eigspec.as_df())
        eigmetrics_df = eigmetrics_df.append(eigspec.metrics_as_df())
        if debug:
            break

    if not debug:
        eigspec_df.to_csv(save_paths["eigspectra"], index=False)
        eigmetrics_df.to_csv(save_paths["eigmetrics"], index=False)


def get_eigenspectrum(
    dataset, data_dir, activations_extractor, pooling, image_transform
):
    if dataset == "imagenet":
        return EigenspectrumImageNet(
            activations_extractor=activations_extractor,
            pooling=pooling,
            image_transform=image_transform,
        )
    elif dataset == "imagenet21k":
        return EigenspectrumImageNet21k(
            data_dir=data_dir,
            activations_extractor=activations_extractor,
            pooling=pooling,
            image_transform=image_transform,
        )
    elif dataset == "object2vec":
        return EigenspectrumObject2Vec(
            data_dir=data_dir,
            activations_extractor=activations_extractor,
            pooling=pooling,
            image_transform=image_transform,
        )
    elif dataset == "majajhong2015":
        return EigenspectrumMajajHong2015(
            activations_extractor=activations_extractor,
            pooling=pooling,
            image_transform=image_transform,
        )
    else:
        raise ValueError(f"Unknown eigenspectrum dataset: {dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and store eigenspectra of models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["imagenet", "imagenet21k", "object2vec", "majajhong2015"],
        help="Dataset of concepts for which to compute the eigenspectrum",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Data directory containing stimuli"
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="max",
        choices=["max", "avg", "none"],
        help="Perform pooling prior to computing the eigenspectrum",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Compute the eigenspectrum on grayscale inputs",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Just run a single model to make sure there are no errors",
    )
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        data_dir=args.data_dir,
        pooling=args.pooling,
        grayscale=args.grayscale,
        debug=args.debug,
    )
