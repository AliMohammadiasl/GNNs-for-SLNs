# 02 MLP Model

## Purpose
This notebook serves as another feature-based baseline, employing a Multi-Layer Perceptron (MLP) for link prediction on SLN data.

## How it Works
Similar to the CNN model, the MLP uses manually calculated graph properties (Jaccard, Adamic-Adar, etc.) as an input vector. It passes this 1D vector of length 6 through multiple dense (Linear) layers with Batch Normalization and ReLU activations, outputting a final sigmoid probability score for link existence.

## Inputs
- Raw edge files from `./data/w_removal_{dataset}`
- Centralized configuration from `utils.py`

## Outputs
- Model Checkpoints: `./results/MLP/{dataset}/models/`
- Confusion Matrices: `./results/MLP/{dataset}/cm_npy/` and `cm_png/`
- Inference metrics (AUC, ACC, BER): `./results/MLP/{dataset}/csv/`

## Execution
Run all cells. Ensure `DEBUG = False` in Cell 2 if you intend to run the exhaustive training loops across all combinations.