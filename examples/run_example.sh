#!/bin/bash

# Activate the SoBo Python virtual environment before doing the predictions!

# define IDs as in the manuscript (Fig. 5)
ids=( "1" "2" "3" "4" "5" "6" )

# define SMILES matching the order in ids array
smiles_list=( "CSc1nc(ccc1C#N)c2ccc(Cl)cc2" "COc1cc(NC(c2c[n]c(F)cc2)=O)ccc1" "Cc1cc(Nc2ccn(n2)c3ccccc3)cc(C)c1OC(=O)OC(C)(C)C" "COc1cc(C2SCCCS2)c(cc1OC)C(=O)c3ccc4OCOc4c3" "COc1nc(C)cnc1NS(=O)(=O)c2cccnc2c3ccc(CC(C)(C)C)cc3" "O=C(Oc1ccc2C(=O)COc2c1)c3cccs3" )

for i in ${!ids[@]}; do
  # extract SMILES
  smiles=${smiles_list[$i]}

  # extract ID
  id=${ids[$i]}

  echo "Predict for SMILES ${smiles} with NAME ${id}"

  # uncomment to do the predictions
  # sobo --smiles $smiles --name $id -c 1
done
