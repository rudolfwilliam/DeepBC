"""Algorithms for solving the mode deep backtracking optimization problem."""

import torch
from torch.linalg import pinv
from utils import convert_vals_ast

def backtrack_linearize(scm, vars_, vals_ast, lambda_=1e3, num_it=50, sparse=False, n_largest=2, 
                        weights=None, const_idxs=None, log=False, log_file=None, verbose=False, eps=0, **us):
    """Backtracking with constraint linearization (recommended). Can be done in a batched fashion.

       :param SCM scm: Structural causal model to be used
       :param list vars_: The antecedent variables
       :param torch.Tensor vals_ast: A tensor of shape (batch_size, len(vars_)) containing the desired antecedent values
       :param float lambda_: The weight of the constraint
       :param int num_it: The number of iterations
       :param bool sparse: Whether to run sparse DeepBC
       :param int n_largest: The number of largest components to be considered in sparse DeepBC. Only considered if sparse=True
       :param dict weights: Optional weights (must be greater than 0) to encourage variable value preservation 
       (large relative value means strong preservation)
       :param list const_idx: The indices of variables to keep constant during optimization (used for sparse DeepBC)
       :param bool log: Whether to log the loss during optimization.
       :param string log_file: filename for logging. Only considered if log=True.
       :param bool verbose: whether to print backtracking loss at each iteration.
       :param float eps: A small number to avoid/reduce numerical instability
       :param dict us: A dictionary containing the factual exogenous values of the SCM variables
       :return dict us_ast: A dictionary containing the counterfactual exogenous values of the SCM variables
    """
    vals_ast = convert_vals_ast(vals_ast)
    # we need to work with flattened us_pr tensor for practical reasons
    us_pr_flat_init = torch.cat([us[key] for key in scm.graph_structure.keys()], dim=1).clone().detach()
    # optimize over these
    us_pr_flat = us_pr_flat_init.clone().detach().requires_grad_()
    if log:
        losses = []
        losses.append(bc_loss(scm, vars_, vals_ast, lambda_, us_pr_flat, us_pr_flat_init, dist_fun='l2'))
    # leave out constant variables
    if const_idxs is not None:
        active_idxs = torch.tensor([i for i in range(us_pr_flat.shape[1]) if i not in const_idxs])
    else:
        # select all
        active_idxs = torch.tensor([i for i in range(us_pr_flat.shape[1])])
    # set weight matrix
    if weights is None:
        # all 1s
        weights_flat = torch.ones(us_pr_flat.shape[1])
    else:
        # repeat according to dim of u and flatten weights
        weights_flat = torch.cat([torch.tensor(weights[w]).repeat(us[w].shape[1]) for w in scm.graph_structure.keys()], dim=0) 
    def decoder_wrapper(us_pr_flat):
        return torch.stack([scm.decode_flat(us_pr_flat)[var].squeeze(1) for var in vars_], dim=1)
    for _ in range(num_it):
        # first order Taylor approximation
        # compute constant
        f0 = decoder_wrapper(us_pr_flat)
        # compute Jacobian (diagonal computation does not seem to be avoidable, see https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571)
        J = torch.diagonal(torch.autograd.functional.jacobian(decoder_wrapper, us_pr_flat), dim1=0, dim2=2).transpose(dim0=0, dim1=2)[:, active_idxs, :]
        # compute linearization
        temp = vals_ast - f0 + torch.bmm(us_pr_flat[:, active_idxs].unsqueeze(1), J).squeeze(1)
        # solve closed form of linearization 
        with torch.no_grad():
            right = (1/lambda_) * torch.diag(weights_flat[active_idxs]) + torch.bmm(J, J.transpose(1, 2)) + eps * torch.eye(J.shape[1])
            left = (1/lambda_) * weights_flat[active_idxs] * us_pr_flat_init[:, active_idxs].unsqueeze(1) + torch.bmm(temp.unsqueeze(1), torch.transpose(J, 1, 2))
            # pseudo-inverse is more stable than inverse
            us_pr_flat[:, active_idxs] = torch.bmm(left, pinv(right)).squeeze(1)
        us_pr_flat = us_pr_flat.clone().detach().requires_grad_()
        if log:
            losses.append(bc_loss(scm, vars_, vals_ast, lambda_, us_pr_flat, us_pr_flat_init, dist_fun='l2'))
        if verbose:
            print(bc_loss(scm, vars_, vals_ast, lambda_, us_pr_flat, us_pr_flat_init, dist_fun='l2').item())
    if sparse:
        # jumps into a recursion
        return sparsify(scm, vars_, vals_ast, us_pr_flat, n_largest=n_largest, log=log, linearize=True, log_file=log_file, **us)
    if log:
        # save losses for plotting
        torch.save(torch.tensor(losses), log_file + '.pt')
    us_ast = unflatten(us_pr_flat.detach(), us, scm)
    return us_ast

def bc_loss(scm, vars_, vals_ast, lambda_, us_pr_flat, us_flat, weights_flat=None, dist_fun='l2'):
    """Compute the loss for the backtracking problem, required for gradient based optimization. 
       Can be done in a batched fashion, but not recommended due to highly variable convergence time and sensitivity for lambda_."""
    if weights_flat is None:
        weights_flat = torch.ones(us_pr_flat.shape[1])
    if dist_fun == 'l2':
        dist = torch.sum(weights_flat*(us_pr_flat - us_flat)**2, dim=1)
    elif dist_fun == 'l1':
        dist = torch.sum(torch.abs(us_pr_flat - us_flat), dim=1)
    constr = torch.sum((torch.stack([scm.decode_flat(us_pr_flat)[var].squeeze(1) for var in vars_], dim=1) - vals_ast)**2, dim=1) * lambda_
    loss = dist + constr
    return loss.sum()

def backtrack_gradient(scm, vars_, vals_ast, lambda_=1e4, num_it=300, sparse=False, n_largest=2, lr=1e-1, dist_fun='l2', 
                       const_idxs=None, log=False, log_file=None, verbose=False, **us):
    """First-order method (Adam) for solving the backtracking problem.

       :param SCM scm: Structural causal model to be used
       :param list vars_: The antecedent variables
       :param torch.Tensor vals_ast: A tensor of shape (batch_size, len(vars_)) containing the desired antecedent values
       :param float lambda_: The weight of the constraint
       :param int num_it: The number of iterations
       :param bool sparse: Whether to run sparse DeepBC
       :param int n_largest: The number of largest components to be considered in sparse DeepBC. Only considered if sparse=True
       :param list const_idx: The indices of variables to keep constant during optimization (used for sparse DeepBC)
       :param bool log: Whether to log the loss during optimization.
       :param string log_file: filename for logging. Only considered if log=True.
       :param bool verbose: whether to print backtracking loss at each iteration.
       :param dict us: A dictionary containing the factual exogenous values of the SCM variables
       :return dict us_ast: A dictionary containing the counterfactual exogenous values of the SCM variables
    """
    vals_ast = convert_vals_ast(vals_ast)
    # we need to work with flattened us tensors for practical reasons
    # keep us_flat fixed
    us_flat = torch.cat([us[val] for val in scm.graph_structure.keys()], dim=1).clone().detach()
    # these are the variables we want to optimize
    us_pr_flat = torch.cat([us[val] for val in scm.graph_structure.keys()], dim=1).clone().detach().requires_grad_() 
    if log:
        losses = []
        losses.append(bc_loss(scm, vars_, vals_ast, lambda_, us_pr_flat, us_pr_flat, dist_fun='l2'))
    optimizer = torch.optim.Adam([us_pr_flat], lr=lr)
    # optimize
    for _ in range(num_it):
        loss = bc_loss(scm, vars_, vals_ast, lambda_, us_pr_flat, us_flat, dist_fun)
        loss.backward()
        # mask out constant variables
        if const_idxs is not None:
            mask = torch.ones_like(us_pr_flat, dtype=torch.float32)
            mask[:, const_idxs] = 0
            us_pr_flat.grad = us_pr_flat.grad * mask 
        optimizer.step()
        optimizer.zero_grad()
        if verbose:
            print(loss.item())
        if log:
            losses.append(loss.item())
    if sparse:
        # jump into a recursion
        return sparsify(scm, vars_, vals_ast, us_pr_flat, n_largest=n_largest, log=log, linearize=False, log_file=log_file, **us)
    if log:
        # save losses for plotting
        torch.save(torch.tensor(losses), log_file + '.pt')
    us_ast = unflatten(us_pr_flat, us, scm)
    return us_ast

def initialize_us_pr(**us):
    us_pr = {}
    for key, u in us.items():
        us_pr[key] = u.detach().requires_grad_()
    return us_pr

def unflatten(us_pr_flat, us_pr, scm):
    """Convert flattened tensor back to dict (and detach)."""
    us_pr_new = {}
    prev = 0
    for key in scm.graph_structure.keys():
        us_pr_new[key] = us_pr_flat[:, prev:(prev + us_pr[key].shape[1])].clone().detach()
        prev += us_pr[key].shape[1]
    return us_pr_new

def sparsify(scm, vars_, vals_ast, us_pr_flat, lambda_=10000, num_it=50, n_largest=2, 
             linearize=True, log=False, log_file=None, **us):
    # only select components of us_pr_flat that have a large deviation from us_flat
    us_flat = torch.cat([us[key] for key in scm.graph_structure.keys()], dim=1).clone().detach()
    _, top_idxs = torch.topk(torch.abs(us_pr_flat - us_flat), n_largest)
    const_idxs = torch.tensor([i for i in range(us_pr_flat.shape[1]) if i not in top_idxs])
    if linearize:
        return backtrack_linearize(scm, vars_, vals_ast, lambda_=lambda_, num_it=num_it, const_idxs=const_idxs, 
                                   sparse=False, log=log, log_file=log_file, **us)
    else:
        return backtrack_gradient(scm, vars_, vals_ast, lambda_=lambda_, num_it=num_it, const_idxs=const_idxs, 
                                  sparse=False, dist_fun='l2', log=log, log_file=log_file, **us)
