# Deep Backtracking Counterfactuals (DeepBC)

Repository for the paper *Deep Backtracking Counterfactuals for Causally Compliant Explanations*.

<p align="center">
<img src="/assets/DeepBC_plot_github.svg" width="500">
</p>

***

## General
To run our code, first make sure to have [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and [git large file storage](https://git-lfs.com/) installed.

Then, copy or clone this repository. Create a conda environment (assuming that you would like to call it `deepbc`):

```
conda env create -n deepbc -f environment.yaml
```

Our code runs on [PyTorch](https://pytorch.org/), [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/) and [normflows](https://github.com/VincentStimper/normalizing-flows).

Before executing any script, make sure to set your working directory to `deepbc` within the `deepbc` repository or just simply navigate there from the root folder by `cd deepbc`. All scripts assume that this is your working directory.

## Code Organization

We utilize a separate directory for each of the data sets `morphomnist` and `celeba`. However, for improved modularity, common functionalities and components are provided at the top level (`optim`, `data` and `scm` folders). The coarse directory structure is given as follows:

```
.
├── FIGURE_GUIDE.md                           # Guide for reproducing figures
├── deepbc
│      ├── optim                              # Mode DeepBC optimization algorithms
│      ├── sampling                           # Stochastic DeepBC via Langevin MC sampling
│      ├── data                               # General data functionality
│      ├── scm                                # General structural causal model classes
│      ├── celeba
│      │      ├── data
│      │      ├── scm                         # Model classes and training scripts (vae, flow)
│      │      │    ├── config                 # Config files for models
│      │      │    ├── scripts
│      │      │    │     ├── train_flows.py   # Train flow models
│      │      │    │     ├── train_vae.py     # Train vae model
│      │      │    │     └── ... 
│      │      │    └── ...
│      │      ├── baselines                   # Baseline models
│      │      ├── visualizations              # Scripts that reproduce figures from the paper
|      |      └── eval                        # Scripts that evaluate different methods
│      ├── morphomnist
│      └── ...
└── ...    
```

The directory structure of `morphomnist` is analogous to that of `celeba`.

Configurations for the individual architectures and algorithms can be found in `config` directories within the respective subdirectories. For instance, the configuration for the celeba VAE architecture can be found in `./celeba/scm/config/vae.json`.

## Figures

You can reproduce all figures from the paper by running the corresponding modules as described in `FIGURE_GUIDE.md`. For instance, if we would like to reproduce Fig. 3, we need to run

```
python -m morphomnist.visualizations.tast_to_iast
```

## Tables

You can reproduce the table from the paper by running the corresponding modules as described in `TABLE_GUIDE.md`. To evaluate the different metods (to obtain the scores), run

```
python -m celeba.eval.evaluate_metrics
```

## Structural Causal Model (SCM) Training

If you would like to retrain the models that are inside of the structural causal models, run the modules `$.scm.scripts.train_flows` and `$.scm.scripts.train_vae`, where `$` must be replaced by either `morphomnist` or `celeba`. E.g., for `morphomnist`, run

```
python -m morphomnist.scm.scripts.train_flows

python -m morphomnist.scm.scripts.train_vae
```

VAEs may be trained either on CPU or (potentially multiple) GPUs. Training the flows on GPU may result in an error. If you would like to work with the newly trained models rather than the old ones, it is important to first delete the old ones that are stored in `./$/scm/trained_models/checkpoints`. All scripts are set up such that they simply take the parameters of any file whose name starts with the according model name.