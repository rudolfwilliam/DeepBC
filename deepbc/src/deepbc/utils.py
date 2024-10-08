"""Some functions that are needed here and there."""

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from json import load
import torch
import argparse

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    return model

def generate_checkpoint_callback(model_name, dir_path):
    checkpoint_callback = ModelCheckpoint(
    dirpath=dir_path,
    filename= model_name + '-{epoch:02d}',
    monitor=None  # Disable monitoring for checkpoint saving
    )
    return checkpoint_callback

def generate_early_stopping_callback(patience=5):
    early_stopping_callback = EarlyStopping(monitor = 'val_loss', min_delta = 0.0, patience=patience, mode = 'min')
    return early_stopping_callback

def flatten_list(list):
     return sum(list, [])
    
def get_config(config_dir, default):
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--name", help="name of the config file to choose", type=str, default=default)
    args = argParser.parse_args()
    config = load(open(config_dir + args.name + ".json", "r"))
    return config

def convert_vals_ast(vals_ast):
    if type(vals_ast) != torch.Tensor:
        if type(vals_ast) == int or type(vals_ast) == float:
            vals_ast = torch.tensor([[vals_ast]], dtype=torch.float32)
        elif type(vals_ast) == list:
            vals_ast = torch.tensor(vals_ast, dtype=torch.float32)
    if len(vals_ast.shape) == 1:
            vals_ast = vals_ast.view(-1, 1)
    return vals_ast
    
def override(f):
    return f

def overload(f):
    return f
    