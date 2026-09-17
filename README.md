# MPNN for Molecular Property Prediction (BBBP)

> Study / prototype — frozen. Single-hop Message Passing Neural Network in Keras + RDKit predicting blood–brain barrier permeability.

## 1. Overview

Port of the Keras MPNN example. Molecules = undirected graphs (atoms → nodes, bonds → edges). Atom/bond featurizers → message passing → readout → MLP → BBBP binary classification on the `BBBP.csv` DeepChem dataset (auto-downloaded).

Known issue (documented in git log `a54ab2a`): single-hop readout bug — messages propagate one hop only. Fixed readout, bug kept documented for study value.

## 2. Architecture

- `AtomFeaturizer` (symbol, valence, H-count, hybridization) + `BondFeaturizer` (type, conjugation, ring).
- `MPNNLayer`: edge network → messages → GRU-style node update (1 step).
- Global pooling → Dense → sigmoid. Loss: binary cross-entropy. Seed 42.

## 3. Repo layout

```
neural_network_mpnn.py  # 398 lines, flat script (runs on import)
README.md
```

No `src/`, `tests/`, manifest, or `LICENSE`.

## 4. Install

```
pip install -r requirements.txt
# tensorflow>=2.16,<2.20, keras>=3,<4, pandas, matplotlib, rdkit-pypi, numpy
```

## 5. Usage

```bash
python neural_network_mpnn.py
```

Downloads `BBBP.csv` via `keras.utils.get_file` to `~/.keras/datasets/`. Trains and plots. Importing the module trains (see §8).

## 6. Expected output

Training curves + validation AUC/accuracy on BBBP scaffold split. No model artifacts saved.

## 7. Limitations / what this is not

- Didactic single-hop MPNN — not competitive with Chemprop / D-MPNN / pretrained GNNs.
- No scaffold split rigor, no hyperparameter search, no calibration.
- `warnings.filterwarnings("ignore")` hides RDKit/TF warnings.
- No tests, no CLI, no reproducibility log.

## 8. Tests

None. Smoke check = full run (~5–15 min CPU/GPU).

## 9. References

- Keras example: https://keras.io/examples/graph/mpnn-molecular-graphs/
- Gilmer et al., Neural Message Passing for Quantum Chemistry: https://arxiv.org/abs/1704.01212
- GNN review: https://arxiv.org/abs/1812.08434
- DeepChem MPNNModel docs

## 10. License

None declared. Study code — all rights reserved by default.
