import torch
import matplotlib.pyplot as plt
from deepbc.optim import backtrack_linearize
from project_files.morphomnist import MmnistSCM

def generate_bcf(img, i, t, val_ast, var='intensity'):
    """Generate backtracking counterfactual values for the given factual variables and counterfactual variable."""
    scm = MmnistSCM(ckpt_path="./project_files/morphomnist/scm/trained_models/checkpoints/")
    scm.eval()
    us = scm.encode(**{"img" : img.view(-1, 1, 28, 28), "thickness" : t.view(-1, 1), "intensity" : i.view(-1, 1)})
    img = scm.decode(**us)["img"].detach().numpy()
    i_ast = torch.tensor(val_ast, dtype=torch.float32)
    us_ast = backtrack_linearize(scm, ['intensity'], i_ast, **us)
    xs_ast = scm.decode(**us_ast)

    return xs_ast


def generate_bcf_plot(img, i, t, val_ast, var='intensity'):
    """Generate a plot of the original image and the image with the counterfactual variables."""
    img_ast, i_ast, t_ast = generate_bcf(img, i, t, val_ast, var=var) 
    fig = plt.figure()
    # use same color scale for both images
    vmin = min(img.min(), img_ast.min())
    vmax = max(img.max(), img_ast.max())
    fig.add_subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title("intensity: " + str(round(i.item(), 2)) + ", thickness: " + str(round(t.item(), 2)))
    fig.add_subplot(1, 2, 2)
    img_ast = torch.squeeze(img_ast).detach().numpy()
    plt.imshow(img_ast, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title("intensity: " + str(round(i_ast.item(), 2)) + ", thickness: " + str(round(t_ast.item(), 2)))
    plt.show()
    