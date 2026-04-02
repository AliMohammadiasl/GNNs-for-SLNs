# 01 CNN Model

## Purpose
This notebook implements a baseline Convolutional Neural Network (CNN) for link prediction within Social Learning Networks (SLNs).

## How it Works
Unlike Graph Neural Networks, CNNs are designed for grid-like data (e.g., images). To apply a CNN to graph structured data, this notebook first calculates several heuristic network metrics for each potential link (Jaccard Coefficient, Adamic-Adar Index, Preferential Attachment, Resource Allocation Index, Shortest Path Length). These 1D feature arrays are expanded into a 2D format to mimic a "1D image" with channels, allowing a `Conv2d` layer to process them. 

## Inputs
- Raw edge removal files from `./data/w_removal_{dataset}`
- Seed lists and percentages from `utils.py`

## Outputs
- Trained PyTorch models: `./results/CNN/{dataset}/models/`
- Confusion Matrices (Numpy & PNG): `./results/CNN/{dataset}/cm_npy/` and `cm_png/`
- CSV Results containing AUC, Accuracy, and Balanced Error Rate: `./results/CNN/{dataset}/csv/`

## Execution
Run all cells. Toggle `DEBUG = True` in Cell 2 to run a fast, single-iteration pipeline for validation.