# Results: base GLM-OCR vs LoRA fine-tuned

Held-out evaluation on 30 CoSyn chemical molecules that were never used for training.

| Metric | Base | LoRA | Change |
|---|---|---|---|
| Exact match (canonical SMILES) | 6.7% | 23.3% | +16.7 pts |
| Valid SMILES | 50.0% | 70.0% | +20.0 pts |
| Mean character similarity | 31.8% | 61.1% | +29.3 pts |

## Per-sample outputs

| # | Target | Base | LoRA |
|---|---|---|---|
| 1 | `CCO.CCCCC(=O)O` | ❌ `CCCC(=O)C(O)C` | ❌ `CCCCC(=O)O` |
| 2 | `CCOC=O` | ✅ `CCOC=O` | ✅ `CCOC=O` |
| 3 | `CC1=CC(=C(C(=C1O)C)OC)C(C=CC2=CC(=C(C(=C2O)C)OC)C(CO)O)O` | ❌ `Cc1cc(C)c(O)c(C)c1OcC(O)c1cc(C)c(O)c1OcC(O)c1cc(C)c(O)c1C` | ❌ `CC1=CC(=C(C=C1C=C/C(=C2C(=C(C=C2C=C1)O)C)O)OC)C(O)CO` |
| 4 | `CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC=CC=C(C)C=CC=C(C)C=CC2=C(CCCC2(C)C)C)C)C` | ❌ `C/C=C/C(C)=C/C=C/C(C)=C/C=C/C(C)=C/C=C/C(C)=C/C=C/C(C)=C/C=C/C(C)=C/C=… (+331 chars)` | ❌ `CC1CCCC(C)(C)C(=C1C)/C=C/C=C/C(C)=C/C=C/C=C/C(C)=C/C=C/C(C)=C/C=C/C(C)… (+33 chars)` |
| 5 | `[Cu+].[N-]=[N+]=[N-]` | ❌ `$$ \mathrm {C u} ^ {+} \mathrm {N} ^ {-} \mathrm {N} ^ {+} \mathrm {N}… (+9 chars)` | ❌ `[N+](=N=N=N)Cu` |
| 6 | `CCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCC)OC(=O)CCCCCCCCCCC` | ❌ `CCCCCCCCCCCCCCCCCCCCC(=O)C(=O)COC(=O)COC(=O)COC(=O)COC(=O)COC(=O)COC(=… (+236 chars)` | ❌ `CCCCCCCCCCCCCCC(=O)OC(COC(=O)CCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCC` |
| 7 | `CC(C)(C#N)CC(C)C=CC(c1ccccc1)CC(C)(C#N)CC(C)C=CC(c2ccccc2)CC(C)(C#N)` | ❌ `C(C)C(C)C(C)C(C)C(C)C(C)C(C)C(C)C(C)C(C)C(C)C(C)C(C)C(C)C(C)C(C)C(C)C(… (+271 chars)` | ❌ `'''rdkit CC(C#N)C(C)C=CC(C#N)CC(C#N)(C)C(C#N)C(C#N)C(C#N)C(C#N)C(C#N)C… (+245 chars)` |
| 8 | `O=C1O[C@H]([C@H](CO)O)C(O)=C1O` | ❌ `C(=O)C(=C)C(=O)[C@@H](O)CO` | ❌ `O=C1C(=C(C(=O)O1)[C@H](CO)O)O` |
| 9 | `CC=O` | ✅ `CC=O` | ✅ `CC=O` |
| 10 | `N[C@@H](CC(=O)N(CCC(=O)O)C)C(=O)O` | ❌ `C(N)CCC(=O)O<sup>+</sup>C[C@@H](C(=O)O)N<sup>+</sup>` | ❌ `CN(CCC(=O)O)C(=O)C[C@H](C(=O)O)N` |
| 11 | `CC(C)(C)C(=O)OCCOC(=O)C(C)(C)C` | ❌ `C(C)(C)C(=O)OCCOC(=O)C(C)(C)C` | ✅ `CC(C)(C)C(=O)OCCOC(=O)C(C)(C)C` |
| 12 | `Oc1ccc(/C=C/c2cc(O)cc(O)c2)cc1` | ❌ `C=Cc1cc(O)cc(O)c1<sep>OH</sep>` | ❌ `O=Cc1ccc(O)cc1O` |
| 13 | `O=Cc1ccccc1` | ❌ `C=OC1=CC=C1` | ✅ `O=Cc1ccccc1` |
| 14 | `CC12CCC(=O)C=C1CCC3C2CCC4(C(=O)CCC34C)C(=O)CO` | ❌ `C(C)(C)C1(C)C2C3C4C(C)(C)C(C(=O)C)C(C)(C)C4C3C2C1C3C4C(C)(C)C(C(=O)C)C… (+221 chars)` | ❌ `CC1(CCC2C3CCC(C2C1=CC(=O)CC3)(C)C)C(=O)CCO` |
| 15 | `[Na+].[Na+].[O-]S(=O)(=O)OS(=O)(=O)[O-]` | ❌ `S(=O)(=O)(=O)S(=O)(=O)O(S(=O)(=O)(=O)S(=O)(=O)(=O)O)` | ❌ `[Na+].[Na+].[O-].[O-].[S-].[S-].[O-].[O-].[O-].[O-]` |
| 16 | `C1=NC(=C2C(=N1)N(C=N2)C3C(C(C(O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N` | ❌ `C(C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)… (+272 chars)` | ❌ `Nc1ncnc2c1ncn2C(O)C(O)C(O)C(O)OP(O)(=O)OP(O)(=O)OP(O)(=O)O` |
| 17 | `[Na+].[Cl-].[O-]Cl` | ❌ `$$ \mathrm {O} ^ {-} \mathrm {C l} \mathrm {N a} ^ {+} $$` | ❌ `[Na+].[O-].[Cl-]` |
| 18 | `CC1=C(C(CCC1)(C)C)C=CC(=C)C=O` | ❌ `C=C(CC=O)C=C/C1=C(CCC1(C)C)C` | ❌ `CC1=C(C(CCC1)(C)C)/C=C/C(=C)C=O` |
| 19 | `C(C1C(C(C(C(O1)OC2(C(C(C(O2)CO)O)O)CO)O)O)O)O` | ❌ `C(C)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(O)C(… (+271 chars)` | ❌ `CC1C(OC(OC2C(C1O)CO)OCC2)OCCO` |
| 20 | `COC(=O)C1=CC=CC=C1O` | ❌ `C(O)C(=O)C1=C=C=C1O` | ✅ `COC(=O)c1ccccc1O` |
| 21 | `[Ag+].[Br-]` | ❌ `<html><body><table><tr><td>$Br^{-}$</td><td>$Ag^{+}$</td></tr><tr><td>… (+43 chars)` | ❌ `[Br]^- [Ag+]` |
| 22 | `CC12CCC(=O)C=C1CCC3C4CCC(C(=O)CO)(C(O)CC34C)C2` | ❌ `C1C2C3C(C2C1C3C(O)C(=O)CO)C(C)(C)C(C)(C)C(C)(C)C(C)(C)C(C)(C)C(C)(C)C(… (+283 chars)` | ❌ `CC12CCC3CC(C12)C(CCC3C(O)C)C(=O)CO` |
| 23 | `CCCCCC` | ❌ `CCCCC` | ❌ `CCCCC` |
| 24 | `O=C1OC(=O)C=C1` | ❌ `C=C C(=O)O C(=O)` | ✅ `O=C1C=CC(=O)O1` |
| 25 | `[Co@OH3]([OH2])([OH2])([OH2])([OH2])([OH2])` | ❌ `$$ \mathrm {C o} \mathrm {O} _ {4} \mathrm {H} _ {2} \mathrm {O} \math… (+49 chars)` | ❌ `[Co]([H2O])([OH])([H2O])[OH]O` |
| 26 | `CN(C)c1cccc2cccc(N(C)C)c12` | ❌ `C(N)(C)C(=C)C1=C2C=CC=C2C(N)(C)C1=C2C=CC=C2C(N)(C)C1=C2C=CC=C2C(N)(C)C… (+233 chars)` | ❌ `CN(C)C1=C2C=CC=C2C(=C1)N(C)C` |
| 27 | `NNc1ccccc1` | ❌ `C=C(N)C=C` | ✅ `NNc1ccccc1` |
| 28 | `C1CO1` | ❌ `Chemical structure of a triene with three carbon atoms and one oxygen … (+5 chars)` | ❌ `O1C2C1=O` |
| 29 | `c1ccc(cc1)O` | ❌ `C(C=C)C=C` | ❌ `Oc1ccc2cccc12` |
| 30 | `CC1CCN2CC(C1)C(C(C2)O)C3=CC=CN4C3=CC=C4` | ❌ `C(C)C1CC2C(C(C(C(C(C2N3C=CC=C3)O)O)O)C=C)C` | ❌ `CC1CC2CN3CC1C(C2C3)C(=CC=C1)C4=CC=CN4` |
