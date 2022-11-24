# Examples from the manuscript

In order to examplify the application of the SoBo model, we take examples as shown in the manuscript (Fig. 5) and run those compounds with the current implementation:

| ID  | SMILES                                             |
| --- | -------------------------------------------------- |
| 1   | CSc1nc(ccc1C#N)c2ccc(Cl)cc2                        |
| 2   | COc1cc(NC(c2c[n]c(F)cc2)=O)ccc1                    |
| 3   | Cc1cc(Nc2ccn(n2)c3ccccc3)cc(C)c1OC(=O)OC(C)(C)C    |
| 4   | COc1cc(C2SCCCS2)c(cc1OC)C(=O)c3ccc4OCOc4c3         |
| 5   | COc1nc(C)cnc1NS(=O)(=O)c2cccnc2c3ccc(CC(C)(C)C)cc3 |
| 6   | O=C(Oc1ccc2C(=O)COc2c1)c3cccs3                     |

Results of predictions are given as `csv` and `svg` files.
For the system with `ID = 1` the total computation time was 8 minutes and 45 seconds on a single CPU (Macbook Pro 1,4 GHz Quad-Core Intel Core i5, 16GB RAM).

## Run a SoBo prediction

After creating a proper Python virtual environment and installing the SoBo model, predictions are obtained as follows, where `SMILES` represents the SMILES string and `NAME` represents the directory in which the predictions will be placed.

```bash
> sobo --smiles <SMILES> --name <NAME>
```

In order to see all possible command-line flags run:

```bash
> sobo --help
```
