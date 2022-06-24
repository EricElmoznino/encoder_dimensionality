# Encoder Dimensionality

Code for a project investigating the effective dimensionality of deep neural networks, and its relationship to their ability to predict neural activity in visual cortex.

---
# Reproducing results and figures in the main manuscript

## Required packages

The code has been tested on Python 3.7.x with the following packages:
- `brain-score` [package link](https://github.com/brain-score/brain-score)
- `model-tools` [package link](https://github.com/brain-score/model-tools)
- `candidate-models` [package link](https://github.com/brain-score/candidate_models)
- `jupyter-lab` (for the figure generation)

The above is not an exaustive list. There are number of additional required packages (e.g. `pytorch`, `tensorflow`, `xarray`), which should be installed automatically as dependencies of those in the above list if you're using `pip`. After that, just try running the scripts following the commands below and install any remaining packages that Python complains are missing.

## Generating results files

To reproduce the results in the main manuscript, the following scripts must be run from the root project directory with the indicated command line arguments.

To compute the effective dimensionalities of all models:
```
python -m scripts.compute_eigenspectra --dataset imagenet
```

To compute the encoding performance of all models:
```
python -m scripts.fit_encoding_models --no_pooling
```

To compute transfer classification performance on ImageNet-21k (please reach out to me to receive the dataset):
```
python -m scripts.compute_nshot --dataset imagenet21k --data_dir [directory to imagenet21k dataset] --classifier prototype
```

To compute model projection distances on ImageNet-21k (please reach out to me to receive the dataset):
```
python -m scripts.compute_projection_distances --dataset imagenet21k --data_dir [directory to imagenet21k dataset]
```

Each of these scripts will generate one or more `*.csv` file in the `results/` directory.

## Generating figures

To reproduce the figures in the main manuscript, run the jupyter notebooks in the `figures/manuscript/` directory. These read from the generated `*.csv` result files. 

Some code blocks generate figures from the Appendix and will rely on results files not created in the previous section (e.g. ZCA-transformed feature results). To generate those Appendix figures as well, run the other relevant scripts in the `scripts/` directory (scroll down to the bottom of the files to see the required arguments). Don't hesitate to reach out to me if you run into issues during this process.
