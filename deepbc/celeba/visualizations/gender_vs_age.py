from celeba.data.datasets import CelebaContinuous
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt
import torch

rg = torch.arange(-3, 3, 0.5)

def main():
    # load data
    data = CelebaContinuous()
    # plot age against gender
    
    res = fisher_exact(table, alternative='greater')
    plt.scatter([data.cont_attr[i][attrs.index("gender")] for i in range(len(data))], [data.cont_attr[i][attrs.index("age")] for i in range(len(data))])

if __name__ == "__main__":
    main()