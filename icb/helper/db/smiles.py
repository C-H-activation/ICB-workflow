# we used those SMILES to train our machine-learning approach
DB = [
    "N(C)1C=CC=C1",  # collab1
    "C1=C(C)C=CC=C1",  # collab2
    "C1=C(CN2C=CC=C2)C=CC=C1",  # collab3
    "C1=CC=NC(C)=C1",  # collab4
    "C1=CC=NC(CN2C=CC=C2)=C1",  # collab5
    "C1=CC=NC(CC2=CC=CC=C2)=C1",  # collab6
    "C1=CC=NC(CCC2=CC=CC=C2)=C1",  # collab7
    "C1=NN=CC=C1",  # collab8
    "N(C)1C=NC=C1",  # collab9
    "C1=CC=CC=C1Cl",  # collab10
    "C1=CC=CC=C1I",  # collab11
    "C1=CC(OCC)=CC=C1Br",  # collab12
    "C1=CC=CC(Br)=C1C(C)C",  # collab13
    "C1=C(CC(C)C)C=CC(Br)=C1",  # collab14
    "C1=CC(C)=CC=C1Cl",  # collab15
    "C1(C(=O)C)=C(OC)C=CC=C1",  # collab16
    "C1=C2OCOC2=CC=C1",  # collab17
    "C1=CC=CC=C1C1=NC=CC=C1",  # collab18
    "C1=CC=NC=C1",  # collab19
    "C1=C(C(=O)OC)C(F)=NC=C1",  # collab20
    "N1=CC=C(C)C=C1C",  # collab21
    "N1=CC(Br)=CC=C1C#N",  # collab22
    "C1C2=C(N=C(S2)C)C=CC=1",  # comp9
    "C1C2=C(N=C(O2)C)C=CC=1",  # comp1
    "C1C2=C(N=C(O2)C)C=C(Br)C=1",  # comp2
    "C1C2=C(N=C(O2)C)C=C(OC)C=1",  # comp5
    "C1C2=C(N=C(O2)C)C(C)=CC=1",  # comp6
    "C1C2=C(N=C(O2)C2C=CC(OC)=CC=2)C=CC=1",  # comp8
    "C1=NC=NC=C1",  # comp13
    "C1C=CN=C(C)N=1",  # comp14
    "C1(C(=O)OC)=CC=C(C#N)C=C1",  # comp33
    "C1OC=CC=1",  # comp41
    "C1SC(C)=CC=1",  # comp43
    "C1SC=CC=1C1=CC=C(C)C=C1",  # comp53
    "C1=CC(F)=CC(Cl)=C1",  # comp163
    "C1=CC(C#N)=CC(C#N)=C1",  # comp171
    "C1=CC(F)=CC=C1Cl",  # comp173
    "C1=CC(C#N)=CC=C1Cl",  # comp175
    "C1=CC(C#N)=NC=C1Br",  # comp177
    "C1=C(F)C(Cl)=NC=C1",  # comp179
    "O1C(C#N)=CC=C1C",  # comp181
    "S1C(C#N)=CC=C1Br",  # comp183
    "N1(C)C(C#N)=CC=C1C",  # comp185
    "S1C(C#N)=CC=C1C",  # comp187
    "S1C=CC(C#N)=C1",  # comp189
    "S1C=CC(Cl)=C1",  # comp191
    "N1(C)N=CC=C1",  # comp193
    "C1C=CC=C2C=COC=12",  # comp195
    "C1C2=C(N=C(N2)C)C=CC=1",  # comp3
    "C1(Cl)=CC=NC2NC(C)=CC1=2",  # comp4
    "C1(=CC=CC2N=C(C)NC1=2)C",  # comp11
    "C1C2=C(N=C(N2)C)C=CC=1F",  # comp12
    "N1C=CC=N1",  # comp16
    "N1C=CC(C)=N1",  # comp17
    "C1=CC=NC2NC=CC1=2",  # comp20
    "C1=C(Br)C=NC2NC=CC1=2",  # comp21
    "C1=CC=NC2NC=C(C1=2)CC",  # comp23
    "C1(Cl)=CC=NC2NC(C)=CC1=2",  # comp24
]

# we used those SMILES to train our machine-learning approach
# including
# training set (range 1-185)
# experimental set (c1-15)
# additional set (s1, s4, s5)
DB2 = [
    "C1C2=C(N=C(O2)C)C=CC=1",
    "C1C2=C(N=C(O2)C)C=C(Br)C=1",
    "C1C2=C(N=C(N2)C)C=CC=1",
    "C1(Cl)=CC=NC2NC(C)=CC1=2",
    "C1C2=C(N=C(O2)C)C=C(OC)C=1",
    "C1C2=C(N=C(O2)C)C(C)=CC=1",
    "C1C2=C(N=C(O2)C2C=CC(C)=CC=2)C=CC=1",
    "C1C2=C(N=C(S2)C)C=CC=1",
    "C1(=CC=CC2N=C(C)NC1=2)C",
    "C1C2=C(N=C(N2)C)C=CC=1F",
    "C1=NC=NC=C1",
    "C1C=CN=C(C)N=1",
    "C1C=CN=C(C#N)N=1",
    "N1C=CC=N1",
    "N1C=CC(C)=N1",
    "N1C=CC(C(F)(F)F)=N1",
    "C1=CC=NC2NC=CC1=2",
    "C1=C(Br)C=NC2NC=CC1=2",
    "C1=CN=CC2NC=CC1=2",
    "C1=CC=NC2NC=C(C1=2)CC",
    "C1=CC=NC2N(C(=O)OC(C)(C)C)C=CC1=2",
    "C1(Cl)=CC=CC=C1Cl",
    "C1=C(C(F)(F)F)C=CC=C1Br",
    "C1(C)=CC=CC=C1",
    "C1(C(=O)C)=C(OC)C=CC=C1",
    "C1(C(=O)OC)=CC=C(C#N)C=C1",
    "C1=C2OCOC2=CC=C1",
    "C1(N)=CC=C(Cl)C=C1",
    "C1=CC=CC=C1C1=NC=CC=C1",
    "N1C=CC=C1",
    "C1SC=CC=1",
    "C1OC=CC=1",
    "C1NC(C)=CC=1",
    "C1SC(C)=CC=1",
    "C1OC(C)=CC=1",
    "C1OC=CC=1C(=O)OC",
    "C1(=CC=C(C#N)N1C)C",
    "C1SC=CC=1C#N",
    "C1SC=CC=1Cl",
    "C1SC=CC=1C1=CC=C(C)C=C1",
    "C1(=CC=C(Br)S1)Cl",
    "C1(=CC=C(C)S1)Cl",
    "C1(=CC=C(I)S1)Cl",
    "C1=CC=CN1C",
    "C1=CC=CN1O[Si](C)(C)C",
    "C1=CC=CN1C(=O)OC(C)(C)C",
    "S1C=CC=C1C1SCCCS1",
    "S1C=CC(C#N)=C1",
    "C1C=CC=C2NC=CC=12",
    "C1C=CC=C2SC=CC=12",
    "C1C=CC=C2OC=CC=12",
    "C1C=CC=C2NC(C3=CC=CC=C3)=CC=12",
    "C1C=CC=C2OC(B3OC(C)(C)C(C)(C)O3)=CC=12",
    "C1C=CC=C2N(O[Si](C)(C)C)C=C(C)C=12",
    "C1C=CC=C2N(O[Si](C)(C)C)C3C=CC(CC)=CC=3C=12",
    "C1C=CC=C2SC=C([Si]([H])(C)C)C=12",
    "C1C=CC=C2N(C(=O)OC(C)(C)C)C=C([Si]([H])(C)C)C=12",
    "C1=CC=NC=C1",
    "C1=CC(F)=NC=C1",
    "C1=C(Cl)C(Cl)=NC=C1",
    "C1=C(C(=O)OC)C(F)=NC=C1",
    "N1=CC=C(C(F)(F)F)C=C1Cl",
    "N1=CC=C(C)C=C1C",
    "N1=CC(Br)=CC=C1C#N",
    "C1=CC=CC=C1C",
    "C1(C)NC(C#N)=CC=1",
    "S1C(Cl)=CC=C1C(=O)CC",
    "O1C(C)=CC=C1C(=O)C",
    "N1(C)C=CC=C1C(=O)N(CCCCCC)CCCCCC",
    "C1C=CC=C2N(C)C=CC=12",
    "C1=CC(C(=O)N(CC)CC)=NC=C1",
    "C1C=C(C)C=C2N=C(C)C=CC=12",
    "C1C=C(C(F)(F)F)C=C2N=C(C)C=CC=12",
    "C1=NC(Cl)=C(C)N=C1",
    "C1C=CC=C2C=CB(C3=CC=C(C(F)(F)F)C=C3)N(C)C=12",
    "C1=CC(F)=CC(Cl)=C1",
    "C1=CC(F)=CC(Br)=C1",
    "C1=CC(F)=CC(I)=C1",
    "C1=CC(F)=NC(=C1)Cl",
    "C1=CC(C#N)=CC(C#N)=C1",
    "C1=CC(F)=CC=C1Cl",
    "C1=CC(C#N)=CC=C1Cl",
    "C1=CC(C#N)=NC=C1Br",
    "C1=C(F)C(Cl)=NC=C1",
    "O1C(C#N)=CC=C1C",
    "S1C(C#N)=CC=C1Br",
    "N1(C)C(C#N)=CC=C1C",
    "S1C(C#N)=CC=C1C",
    "N(C)1C=CC=C1",
    "C1=C(C)C=CC=C1",
    "C1=C(CN2C=CC=C2)C=CC=C1",
    "C1=CC=NC(C)=C1",
    "C1=CC=NC(CN2C=CC=C2)=C1",
    "C1=CC=NC(CC2=CC=CC=C2)=C1",
    "C1=CC=NC(CCC2=CC=CC=C2)=C1",
    "C1=NN=CC=C1",
    "N(C)1C=NC=C1",
    "C1=CC=CC=C1Cl",
    "C1=CC=CC=C1I",
    "C1=CC(OCC)=CC=C1Br",
    "C1=CC=CC(Br)=C1C(C)C",
    "C1=C(CC(C)C)C=CC(Br)=C1",
    "C1=CC(C)=CC=C1Cl",
    "C1=CC=CC=C1OC",
    "C1=CC(N2C=C(C)C(C)=C2C)=CC(Cl)=C1",
    "C1=CC=CC=C1N(=O)=O",
]