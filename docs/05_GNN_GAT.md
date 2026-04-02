# 05 GNN GAT Model

## Purpose
This notebook implements a Graph Attention Network (GAT) model as an alternative baseline for predicting links. 

## How it Works
Unlike GraphSAGE or GCN which treat neighborhood aggregations relatively uniformly, GAT assigns learned attention weights to different neighbors, enabling the model to focus on specific student connections over others when computing the node's new embedding.

## Inputs
- Raw edge matrices from `./data/w_removal_{dataset}`
- `dgl` package installed and configured.

## Outputs
- Checkpoints: `./results/GAT/{dataset}/models/`
- Confusion Matrices: `./results/GAT/{dataset}/cm_npy/` and `cm_png/`
- Metrics: `./results/GAT/{dataset}/csv/`

## Execution
Run sequentially. Note that the multi-head attention mechanism present in GAT may require significantly more computational resources compared to GCN or MLP.