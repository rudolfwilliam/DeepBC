# Deep Backtracking Counterfactuals (DeepBC)

Repository for the paper *Causally Compliant Counterfactuals for Deep Strucutral Causal Models via Backtracking*

<p align="center">
<img src="/assets/DeepBC_plot_github.svg" width="500">
</p>

***

## General
To run our code, first copy or clone this repository. Then, create a conda environment (assuming that you would like to call it `deepbc`):

```
conda create --name deepbc python=3.11.3
```

After navigating into the root folder of this repository using your terminal, install all necessary packages from the `requirements.txt` as follows (make sure your new conda environment is activated before you run this command):

```
conda install --file requirements.txt
```

Our code runs on [PyTorch](https://pytorch.org/), [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/) and [normflows](https://github.com/VincentStimper/normalizing-flows).

Before executing any script, make sure to set your working directory to `deepbc` within the `deepbc` repository or just simply navigate there from the root folder by `cd deepbc`. All scripts assume that this is your working directory.

## Code Organization

We utilize a separate directory for each of the data sets `morphomnist` and `celeba`. However, for improved modularity, common functionalities and components are provided at the top level (`optim`, `data` and `scm` folders). The coarse directory structure is given as follows:

```
.
├── deepbc
│      ├── optim                              # DeepBC optimization algorithms
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
│      │      └── visualizations              # Scripts that reproduce figures from the paper
│      ├── morphomnist
│      └── ...
└── ...    
```

The coarse directory structure of `morphomnist` is analogous to that of `celeba`.

Configurations for the individual architectures and algorithms can be found in `config` directories within the respective subdirectories. For instance, the configuration for the celeba VAE architecture can be found in `./celeba/scm/config/vae.json`.

## Visualizations

```
python -m morphomnist.visualizations.tast_to_iast
```

## Structural Causal Model (SCM) Training

If you would like to retrain the models that are inside of the structural causal models, run the modules `$.scm.scripts.train_flows` and `$.scm.scripts.train_vae`, where `$` must be replaced by either `morphomnist` or `celeba`. E.g., for `morphomnist`, run

```
python -m morphomnist.scm.scripts.train_flows

python -m morphomnist.scm.scripts.train_vae
```

VAEs may be trained either on CPU or (potentially multiple) GPUs. Training the flows on GPU may result in an error. If you would like to work with the newly trained models rather than the old ones, it is important to first delete the old ones that are stored in `./$/scm/trained_models/checkpoints`. All scripts are set up such that they simply take the parameters of any file whose name starts with the according model name.