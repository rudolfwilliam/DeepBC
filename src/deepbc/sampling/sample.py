"""Stochastic DeepBC."""

from ..optim import *
from ..utils import convert_vals_ast
import torch

def langevin_mc(scm, vars_, vals_ast, lambda_init=1e3, num_it_init=30, lambda_samp=1e4, gap=500, num_samp=100, sparse=False, n_largest=2, 
                weights=None, const_idxs=None, log=False, log_file=None, dist_fun="l2", step_size=1e-6, verbose=False, **us):
    """Perform Langevin Monte Carlo on the SCM for sampling, with initial guess given by optimization.

    :param SCM scm: Structural causal model to be used
    :param list vars_: The antecedent variables:param scm: SCM
    :param torch.Tensor vals_ast: A tensor of shape (batch_size, len(vars_)) containing the desired antecedent values
    :param float lambda_init: The weight of the constraint for the initial guess
    :param int num_it_init: The number of iterations for the initial guess
    :param float lambda_samp: The weight of the constraint for sampling
    :param int gap: The number of iterations between samples
    :param int num_samp: The number of samples to be drawn
    :param bool sparse: Whether to run sparse DeepBC
    :param int n_largest: The number of largest components to be considered in sparse DeepBC. Only considered if sparse=True.
    :param list weights: The weights of the variables in the loss function. Only considered if sparse=True.
    :param list const_idx: The indices of variables to keep constant during optimization (used for sparse DeepBC)
    :param bool log: Whether to log the loss during optimization.
    :param string log_file: filename for logging. Only considered if log=True.
    :param string dist_fun: The distance function to be used in the loss function
    :param float step_size: The step size for Langevin MC
    :param bool verbose: whether to print backtracking loss at each iteration.
    :param dict us: A dictionary containing the factual exogenous values of the SCM variables
    :return dict us_ast: A list containing the counterfactual exogenous values of the SCM variables
    """
    # convert vals_ast to the desired format
    vals_ast = convert_vals_ast(vals_ast)
    # obtain initial guess from linearization
    us_ast_init = backtrack_linearize(scm, vars_, vals_ast, lambda_=lambda_init, num_it=num_it_init, sparse=sparse, 
                                      n_largest=n_largest, weights=weights, const_idxs=const_idxs, log=log, 
                                      log_file=log_file, verbose=verbose, **us)
    # perform Langevin MC (similar to stochastic gradient descent)
    us_flat = torch.cat([us[key] for key in scm.graph_structure.keys()], dim=1).detach()
    us_flat.requires_grad = False
    samples = []
    # these are the variables we want to sample from
    us_pr_flat = torch.cat([us_ast_init[val] for val in scm.graph_structure.keys()], dim=1).detach().requires_grad_()
    # flatten weights
    weights_flat = torch.cat([torch.tensor(weights[key]).repeat(us[key].shape[1]) for key in scm.graph_structure.keys()], dim=0)
    for _ in range(num_samp):
        for _ in range(gap): 
            loss = bc_loss(scm, vars_, vals_ast, lambda_samp, us_pr_flat, us_flat, dist_fun=dist_fun, weights_flat=weights_flat)
            loss.backward()
            # mask out constant variables
            if const_idxs is not None:
                mask = torch.ones_like(us_pr_flat, dtype=torch.float32)
                mask[:, const_idxs] = 0
                us_pr_flat.grad = us_pr_flat.grad * mask
            with torch.no_grad():
                noise = torch.randn_like(us_pr_flat) * torch.sqrt(step_size * torch.tensor(2., dtype=torch.float32))
                us_pr_flat -= step_size * us_pr_flat.grad + noise
            us_pr_flat.grad.zero_()
        samples.append(unflatten(us_pr_flat.detach(), us, scm))
        us_pr_flat = torch.cat([us_ast_init[val] for val in scm.graph_structure.keys()], dim=1).detach().requires_grad_()
    return samples
