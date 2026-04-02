# 04 GNN GCN Model

## Purpose
This notebook implements a Graph Convolutional Network (GCN) model to serve as a baseline comparison for link prediction over the SLN graphs.

## How it Works
Using DGL, the notebook constructs a sum-pooling GCN. It operates on the full graph topology, updating node features iteratively by multiplying them with the graph's adjacency matrix and a learned weight matrix. 

## Inputs
- Graph edges from `./data/w_removal_{dataset}`
- Requires the `dgl` library.

## Outputs
- Checkpoints: `./results/GCN/{dataset}/models/`
- Confusion Matrices: `./results/GCN/{dataset}/cm_npy/` and `cm_png/`
- Inference metrics: `./results/GCN/{dataset}/csv/`

## Execution
Run all cells. As with GraphSAGE, this requires the DGL library and could utilize substantial memory depending on the graph size.