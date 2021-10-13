import os
from PIL import Image
from tqdm import tqdm
from typing import List, Callable


class ImageDatasetTransformer:

    def __init__(self, name: str, transform: Callable[[Image.Image], Image.Image], n_samples: int = 1):
        self.name = name
        self.transform = transform
        self.n_samples = n_samples

    def transform_dataset(self, dataset_name: str, image_paths: List[str]) -> List[str]:
        image_names = [path.split('/')[-1] for path in image_paths]
        framework_home = os.path.expanduser(os.getenv('RESULTCACHING_HOME', '~/.result_caching'))
        save_dir = f'transformed_datasets/dataset={dataset_name},transform={self.name},n_samples={self.n_samples}'
        save_dir = os.path.join(framework_home, save_dir)

        # Create and cache transformed dataset, if it doesn't already exist
        if not os.path.isdir(save_dir):
            print(f'Transforming dataset: {dataset_name} using transform: {self.name}')
            os.makedirs(save_dir)
            for image_path, image_name in tqdm(zip(image_paths, image_names)):
                image = Image.open(image_path)
                for i in range(self.n_samples):
                    transformed_path = os.path.join(save_dir, image_name.replace('.', f'_sample={i:05d}.'))
                    image = self.transform(image)
                    image.save(transformed_path)

        # Retrieve transformed image paths
        transformed_paths = []
        for image_name in image_names:
            transformed_paths += [os.path.join(save_dir, image_name.replace('.', f'_sample={i:05d}.'))
                                  for i in range(self.n_samples)]

        return transformed_paths
