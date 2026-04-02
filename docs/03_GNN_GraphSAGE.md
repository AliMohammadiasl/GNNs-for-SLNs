# 03 GNN GraphSAGE Model

## Purpose
This is the primary Graph Neural Network (GNN) methodology described in the paper. It leverages GraphSAGE (SAmple and aggreGatE) for inductive representation learning to predict link formation over dynamically evolving SLNs.

## How it Works
Instead of manually engineering features like the CNN or MLP, GraphSAGE natively learns node embeddings by aggregating feature information from a node's local neighborhood. The notebook:
1. Loads the PyTorch/DGL environment.
2. Formats the data as a `dgl.graph` object.
3. Implements a 2-layer `SAGEConv` architecture coupled with a `DotPredictor` to compute the inner product of two node embeddings.
4. Uses a weighted binary cross-entropy loss to counteract the extreme sparsity of positive links in SLNs.

## Inputs
- Graph edges from `./data/w_removal_{dataset}`
- Requires the `dgl` (Deep Graph Library) library to execute.

## Outputs
- Checkpoints: `./results/GraphSAGE/{dataset}/models/`
- Confusion Matrices: `./results/GraphSAGE/{dataset}/cm_npy/` and `cm_png/`
- Metrics Dataframes: `./results/GraphSAGE/{dataset}/csv/`

## Execution
Execute the cells sequentially. Because GNNs operate on whole-graph topology, memory usage can scale with dataset size. Ensure `dgl` is properly installed for your hardware (CPU vs. CUDA).