The figures of the paper can be reproduced via the following commands (assuming you working directory is `deepbc` and you have followed the instructions of the `README.md`):

Minor variations in results may arise from hardware disparities. It is anticipated that these discrepancies will remain negligible[^1].

## Morpho-MNIST

Fig. 3: `python -m morphomnist.visualizations.imgs_intensity`

Fig. 4 (a): `python -m morphomnist.visualizations.uiast_to_utast`

Fig. 4 (b): `python -m morphomnist.visualizations.iast_to_tast`

Fig. 4 (c): `python -m morphomnist.visualizations.tast_to_iast`

## CelebA

Fig. 5: `python -m celeba.visualizations.antecedent_beard_sparse`

## Supplementary

Fig. 6:  `python -m morphomnist.visualizations.lin_vs_GD`

Fig. 7: `python -m morphomnist.visualizations.iast_to_tast` (run three times with the reported thickness weights)

Fig. 8 (a): `python -m celeba.visualizations.antecedent_beard_OOD`

Fig. 8 (b): `python -m celeba.visualizations.multi_antecedents`

Fig. 9 (c): `python -m celeba.visualizations.antecedent_beard_suppl`

Fig. 9 (d): `python -m celeba.visualizations.comp_nc_suppl`

Fig. 9 (e): `python -m celeba.visualizations.antecedent_beard_sparse`

## Other figures

*All the figures from the paper that are not listed here were created manually.*

[^1]: Our experiments involved testing on various hardware environments.