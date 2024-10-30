# Deep Backtracking Counterfactuals (DeepBC)

[![Python 3.11.3](https://img.shields.io/badge/python-3.11.3-blue.svg)](https://www.python.org/downloads/release/python-3113/)

Repository for the TMLR paper [*Deep Backtracking Counterfactuals for Causally Compliant Explanations*](https://openreview.net/pdf?id=Br5esc2CXR) by Klaus-Rudolf Kladny, Julius von Kügelgen, Bernhard Schölkopf and Michael Muehlebach.

<p align="center">
<img src="/assets/DeepBC_plot_github.svg" width="500">
</p>

Our code runs on [PyTorch](https://pytorch.org/), [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/) and [normflows](https://github.com/VincentStimper/normalizing-flows).
***

## Citation

If you find our code useful, we would be happy if you could leave our repository a star :star: and cite our publication :page_facing_up:. The bibtex entry is

```bibtex
@article{kladny2024deep,
    title={Deep Backtracking Counterfactuals for Causally Compliant Explanations},
    author={Klaus-Rudolf Kladny and Julius von K{\"u}gelgen and Bernhard Sch{\"o}lkopf and Michael Muehlebach},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2024},
    url={https://openreview.net/forum?id=Br5esc2CXR}
}
```

## Getting Started :rocket:
First, copy or clone this repository. If you only want to access the core functionality of `deepbc` (i.e., in your own project), you can directly jump to the [package installation](#installing-the-`deepbc`-package).
### Reproducing experiments
If you want to reproduce the experiments of our paper, first make sure to have [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and [git large file storage](https://git-lfs.com/) installed.

Create a conda environment (assuming that you would like to call it `deepbc`):

```console
conda env create -n deepbc -f environment.yaml
```

and activate this environment
```console
conda activate deepbc
```
### Installing the `deepbc` package
Install the `deepbc` package (via `setup.py`):
```console
pip install -e .
```
This way, you can simply import required classes and functions from our project. For example, you can run
```python
from deepbc.optim import backtrack_linearize
```
to access the linearization algorithm for mode DeepBC.

## Code Organization

We utilize a separate directory for each of the data sets `morphomnist` and `celeba`. However, for improved modularity, common functionalities and components are provided at the top level (`optim`, `data` and `scm` folders). The coarse directory structure is given as follows:

```
.
├── FIGURE_GUIDE.md                           # Guide for reproducing figures
├── setup.py                                  # Can be used to install core functionality
├── src
│    └─ deepbc                                # Can be installed as package via setup.py
│         ├── optim                           # Mode DeepBC optimization algorithms
│         ├── sampling                        # Stochastic DeepBC via Langevin MC sampling
│         ├── data                            # General data functionality
│         └── scm                             # General structural causal model classes
├── project_files
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
│      │      └── eval                        # Scripts that evaluate different methods
│      ├── morphomnist
│      └── ...
└── ...    
```

The directory structure of `morphomnist` is analogous to that of `celeba`.

Configurations for the individual architectures and algorithms can be found in `config` directories within the respective subdirectories. For instance, the configuration for the celeba VAE architecture can be found in `./project_files/celeba/scm/config/vae.json`.

## Figures

You can reproduce all figures from the paper by running the corresponding modules as described in `FIGURE_GUIDE.md`. For instance, if we would like to reproduce Fig. 3, we need to run

```console
python -m project_files.morphomnist.visualizations.tast_to_iast
```

## Tables

You can reproduce the table from the paper by running the corresponding modules as described in `TABLE_GUIDE.md`. To evaluate the different metods (to obtain the scores), run

```console
python -m project_files.celeba.eval.evaluate_metrics
```

## Structural Causal Model (SCM) Training

If you would like to retrain the models that are inside of the structural causal models, run the modules `project_files.$.scm.scripts.train_flows` and `project_files.$.scm.scripts.train_vae`, where `$` must be replaced by either `morphomnist` or `celeba`. E.g., for `morphomnist`, run

```console
python -m project_files.morphomnist.scm.scripts.train_flows

python -m project_files.morphomnist.scm.scripts.train_vae
```

VAEs may be trained either on CPU or (potentially multiple) GPUs. Training the flows on GPU may result in an error.

>[!IMPORTANT]
>If you would like to work with the newly trained models rather than the old ones, it is important to first delete the old ones that are stored in `./project_files/$/scm/trained_models/checkpoints`. All scripts are set up such that they simply take the parameters of any file whose name starts with the according model name.

## Issues?

If you experience difficulties with the code, just open an issue or write an e-mail to *kkladny [at] tuebingen [dot] mpg [dot] de*. Also, we are happy to merge your pull request if you have an interesting extension/application.
