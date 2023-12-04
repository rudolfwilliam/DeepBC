from optim import *
import torch

def langevin_mc(scm, vars_, vals_ast, lambda_init=1e3, num_it_init=30, lambda_samp=1e4, gap=500, num_samp=100, sparse=False, n_largest=2, 
                weights=None, const_idxs=None, log=False, log_file=None, dist_fun="l2", step_size=1e-6, verbose=False, **us):
    """Perform Langevin Monte Carlo on the SCM for sampling, with initial guess given by optimization."""
    # obtain initial guess from linearization
    us_ast_init = backtrack_linearize(scm, vars_, vals_ast, lambda_=lambda_init, num_it=num_it_init, sparse=sparse, 
                                      n_largest=n_largest, weights=weights, const_idxs=const_idxs, log=log, 
                                      log_file=log_file, verbose=verbose, **us)
    # perform Langevin MC (similar to stochastic gradient descent)
    us_flat = torch.cat([us[val] for val in scm.graph_structure.keys()], dim=1).detach()
    us_flat.requires_grad = False
    samples = []
    # these are the variables we want to sample from
    us_pr_flat = torch.cat([us_ast_init[val] for val in scm.graph_structure.keys()], dim=1).detach().requires_grad_()
    for _ in range(num_samp):
        for _ in range(gap):
            loss = bc_loss(scm, vars_, vals_ast, lambda_samp, us_pr_flat, us_flat, dist_fun)
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
