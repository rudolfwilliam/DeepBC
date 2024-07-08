# Deep Backtracking Counterfactuals (DeepBC)

[![Python 3.11.3](https://img.shields.io/badge/python-3.11.3-blue.svg)](https://www.python.org/downloads/release/python-3113/)

Repository for the paper [*Deep Backtracking Counterfactuals for Causally Compliant Explanations*](https://arxiv.org/pdf/2310.07665).

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

## Code Organization

We utilize a separate directory for each of the data sets `morphomnist` and `celeba`. However, for improved modularity, common functionalities and components are provided at the top level (`optim`, `data` and `scm` folders). The coarse directory structure is given as follows:

```
.
├── FIGURE_GUIDE.md                           # Guide for reproducing figures
├── deepbc
|      ├── src
│      │    ├── optim                         # Mode DeepBC optimization algorithms
│      │    ├── sampling                      # Stochastic DeepBC via Langevin MC sampling
│      │    ├── data                          # General data functionality
│      │    └── scm                           # General structural causal model classes
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
python -m deepbc.morphomnist.visualizations.tast_to_iast
```

## Tables

You can reproduce the table from the paper by running the corresponding modules as described in `TABLE_GUIDE.md`. To evaluate the different metods (to obtain the scores), run

```
python -m deepbc.celeba.eval.evaluate_metrics
```

## Structural Causal Model (SCM) Training

If you would like to retrain the models that are inside of the structural causal models, run the modules `deepbc.$.scm.scripts.train_flows` and `deepbc.$.scm.scripts.train_vae`, where `$` must be replaced by either `morphomnist` or `celeba`. E.g., for `morphomnist`, run

```
python -m deepbc.morphomnist.scm.scripts.train_flows

python -m deepbc.morphomnist.scm.scripts.train_vae
```

VAEs may be trained either on CPU or (potentially multiple) GPUs. Training the flows on GPU may result in an error. If you would like to work with the newly trained models rather than the old ones, it is important to first delete the old ones that are stored in `./deepbc/$/scm/trained_models/checkpoints`. All scripts are set up such that they simply take the parameters of any file whose name starts with the according model name.

## Citation

If you find our code useful, we would be happy if you could leave our repository a star and cite our preprint. The bibtex entry is

```biblatex
@article{kladny2023deep,
  title={Deep backtracking counterfactuals for causally compliant explanations},
  author={Kladny, Klaus-Rudolf and von K{\"u}gelgen, Julius and Sch{\"o}lkopf, Bernhard and Muehlebach, Michael},
  journal={arXiv preprint arXiv:2310.07665},
  year={2023}
}
```

## Issues?

If you experience difficulties with the code, just open an issue or write an e-mail to *kkladny [at] tuebingen [dot] mpg [dot] de*. Also, we are happy to merge your pull requests if you have an interesting extension.
