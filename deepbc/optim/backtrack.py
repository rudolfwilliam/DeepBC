"""Algorithms for solving the deep backtracking optimization problem."""

import torch

def backtrack_linearize(scm, vars_, vals_ast, lambda_=1e5, num_it=50, sparse=False, n_largest=2, const_idxs=None, **us):
    """Backtracking with constraint linearization (recommended). Can be done in a batched fashion.

       :param SCM scm: Structural causal model to be used
       :param list vars_: The antecedent variables
       :param torch.Tensor vals_ast: A tensor of shape (batch_size, len(vars_)) containing the desired antecedent values
       :param float lambda_: The weight of the constraint
       :param int num_it: The number of iterations
       :param bool sparse: Whether to run sparse DeepBC
       :param int n_largest: The number of largest components to be considered in sparse DeepBC. Only considered if sparse=True.
       :param list const_idx: The indices of variables to keep constant during optimization
       :param dict us: A dictionary containing the factual exogenous values of the SCM variables
       :return dict us_ast: A dictionary containing the counterfactual exogenous values of the SCM variables
    """
    us_pr = initialize_us_pr(**us)
    # we need to work with flattened us_pr tensor for practical reasons
    us_pr_flat_init = torch.cat([u for u in us_pr.values()], dim=1)
    # optimize over these
    us_pr_flat = us_pr_flat_init.clone().detach().requires_grad_()
    # leave out constant variables
    if const_idxs is not None:
        active_idxs = torch.tensor([i for i in range(us_pr_flat.shape[1]) if i not in const_idxs])
    else:
        # select all
        active_idxs = torch.tensor([i for i in range(us_pr_flat.shape[1])])
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
        # solve closed form for linearization 
        with torch.no_grad():
            right = torch.eye(us_pr_flat[:, active_idxs].shape[1]) + lambda_ * torch.bmm(J, J.transpose(1, 2))
            left = us_pr_flat_init[:, active_idxs].unsqueeze(1) + lambda_ * torch.bmm(temp.unsqueeze(1), torch.transpose(J, 1, 2))
            us_pr_flat[:, active_idxs] = torch.bmm(left, torch.inverse(right)).squeeze(1)
        us_pr_flat = us_pr_flat.clone().detach().requires_grad_()
    if sparse:
        # jumps into a recursion
        return sparsify(scm, vars_, vals_ast, us_pr_flat, n_largest=n_largest, **us)
    us_ast = unflatten(us_pr_flat.detach(), us_pr)
    return us_ast

def bc_loss(scm, vars_, vals_ast, lambda_, us_pr_flat, us_flat, dist_fun='l1'):
    """Compute the loss for the backtracking problem, required for gradient based optimization. 
       Can be done in a batched fashion, but not recommended due to highly variable convergence time and sensitivity for lambda_."""
    if dist_fun == 'l2':
        dist = torch.sum((us_pr_flat - us_flat)**2, dim=1)
    elif dist_fun == 'l1':
        dist = torch.sum(torch.abs(us_pr_flat - us_flat), dim=1)
    constr = torch.sum((torch.stack([scm.decode_flat(us_pr_flat)[var].squeeze(1) for var in vars_], dim=1) - vals_ast)**2, dim=1) * lambda_
    loss = dist + constr
    return loss.sum()

def backtrack_gradient(scm, vars_, vals_ast, lambda_=10000, num_it=30000, lr=1e-6, dist_fun='l2', const_idxs=None, **us):
    """Simple gradient descent method for solving the backtracking problem (not recommended, can be unstable)"""
    # initialize "us prime"
    us_pr = initialize_us_pr(**us)
    # we need to work with flattened us tensors for practical reasons
    # keep us_flat fixed
    us_flat = torch.cat([u for u in us.values()], dim=1).detach()
    us_flat.requires_grad = False
    # these are the variables we want to optimize
    us_pr_flat = torch.cat([u for u in us_pr.values()], dim=1).detach().requires_grad_() 
    # optimize
    for i in range(num_it):
        loss = bc_loss(scm, vars_, vals_ast, lambda_, us_pr_flat, us_flat, dist_fun)
        loss.backward()
        # mask out constant variables
        if const_idxs is not None:
            mask = torch.ones_like(us_pr_flat, dtype=torch.float32)
            mask[:, const_idxs] = 0
            us_pr_flat.grad = us_pr_flat.grad * mask 
        with torch.no_grad():
            us_pr_flat = us_pr_flat - lr*us_pr_flat.grad
        us_pr_flat.requires_grad = True
        if i % 10 == 0:
            print(f"loss: {loss.item()}")
    us_ast = unflatten(us_pr_flat.detach(), us_pr)
    return us_ast

def initialize_us_pr(**us):
    us_pr = {}
    for key, u in us.items():
        us_pr[key] = torch.clone(u).requires_grad_()
    return us_pr

def unflatten(us_pr_flat, us_pr):
    """Convert flattened tensor back to dict (and detach)."""
    us_pr_new = {}
    prev = 0
    for key, u in us_pr.items():
        us_pr_new[key] = us_pr_flat[:, prev:(prev + u.shape[1])]
        prev += u.shape[1]
    return us_pr_new

def sparsify(scm, vars_, vals_ast, us_pr_flat, lambda_=10000, num_it=30, n_largest=2, linearize=True, **us):
    # only select components of us_pr_flat that have a large deviation from us_flat
    us_flat = torch.cat([u for u in us.values()], dim=1).detach()
    _, top_idxs = torch.topk(torch.abs(us_pr_flat - us_flat), n_largest)
    const_idxs = torch.tensor([i for i in range(us_pr_flat.shape[1]) if i not in top_idxs])
    if linearize:
        return backtrack_linearize(scm, vars_, vals_ast, lambda_=lambda_, num_it=num_it, const_idxs=const_idxs, sparse=False, **us)
    else:
        return backtrack_gradient(scm, vars_, vals_ast, lambda_=lambda_, num_it=num_it, const_idxs=const_idxs, dist_fun='l2', **us)
