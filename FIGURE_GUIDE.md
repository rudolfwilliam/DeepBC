The figures of the paper can be reproduced via the following commands (assuming you working directory is the top level directory of the repository and you have followed the instructions of the `README.md`):

Minor variations in results may arise from hardware disparities. It is anticipated that these discrepancies will remain negligible[^1].

## Morpho-MNIST

Fig. 4 (i): `python -m project_files.morphomnist.visualizations.uiast_to_utast`

Fig. 4 (ii): `python -m project_files.morphomnist.visualizations.iast_to_tast`

Fig. 4 (iii): `python -m project_files.morphomnist.visualizations.tast_to_iast`

Fig. 5: `python -m project_files.morphomnist.visualizations.imgs_intensity`

## CelebA

Fig. 6: `python -m project_files.celeba.visualizations.antecedent_beard_sparse`

Fig. 7: `python -m project_files.celeba.visualizations.antecedent_beard_OOD`

Fig. 8: `python -m project_files.celeba.visualizations.sample`

## Supplementary

Fig. 9:  `python -m project_files.morphomnist.visualizations.lin_vs_GD`

Fig. 10: `python -m project_files.morphomnist.visualizations.sample_iast_to_tast`

Fig. 11: `python -m project_files.morphomnist.visualizations.iast_to_tast` (run three times with the reported thickness weights)

Fig. 12 (a): `python -m project_files.celeba.visualizations.multi_antecedents`

Fig. 12 (b): `python -m project_files.celeba.visualizations.antecedent_beard_suppl`

Fig. 12 (c): `python -m project_files.celeba.visualizations.antecedent_beard_sparse`

## Other figures

*All the figures from the paper that are not listed here were created manually.*

[^1]: Our experiments involved testing on various hardware environments.