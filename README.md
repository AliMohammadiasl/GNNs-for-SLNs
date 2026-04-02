# Spatiotemporal Link Formation Prediction in Social Learning Networks Using Graph Neural Networks

## Abstract
Social learning networks (SLNs) are graphical representations that capture student interactions within educational settings (e.g., a classroom), with nodes representing students and edges denoting interactions. Accurately predicting future interactions in these networks (i.e., link prediction) is crucial for enabling effective collaborative learning, supporting timely instructional interventions, and informing the design of effective group-based learning activities. However, traditional link prediction approaches are typically tuned to general online social networks (OSNs), often overlooking the complex, non-Euclidean, and dynamically evolving structure of SLNs, thus limiting their effectiveness in educational settings. 

In this work, we propose a graph neural network (GNN) framework that jointly considers the temporal evolution within classrooms and spatial aggregation across classrooms to perform link prediction in SLNs. 

## Key Findings
Based on our empirical analysis over multiple SLNs (e.g., Virtual Shakespeare, Machine Learning, Algorithms, English Composition):
- **Hypothesis 1 (Temporal Domain):** SLN's temporal structures have a significant impact on GNNs' prediction capabilities. We observe statistically significant performance improvements (AUC) in the prediction of future links as the courses progress temporally.
- **Hypothesis 2 (Spatial Domain):** Aggregating SLNs from multiple classrooms generally enhances model performance equitably, especially in sparser datasets (e.g., Virtual Shakespeare and English Composition) where the SLN topology benefits most from combined structural data.
- **Hypothesis 3 (Spatiotemporal Analysis):** Jointly leveraging both the temporal evolution and spatial aggregation of SLNs significantly outperforms conventional baseline approaches that analyze classrooms in isolation. Combining SLNs earlier in the course yields the strongest performance gains compared to the state-of-the-art.

## Repository Structure
The repository contains the data, utilities, and Jupyter notebooks to reproduce the findings:
```text
├── 01_CNN_Model.ipynb         # Baseline CNN model
├── 02_MLP_Model.ipynb         # Baseline MLP model
├── 03_GNN_GraphSAGE.ipynb     # GraphSAGE model (Primary GNN)
├── 04_GNN_GCN.ipynb           # Baseline GCN model
├── 05_GNN_GAT.ipynb           # Baseline GAT model
├── 06_Post_Processing.ipynb   # Aggregates inference metrics
├── 07_Data_Analysis.ipynb     # Calculates graph topological metrics
├── utils.py                   # Shared data loaders, metric calculators, and configs
├── data/                      # Raw SLN edge lists and metadata
├── docs/                      # Documentation for each individual notebook
```

For detailed explanations of each notebook, please refer to the `docs/` directory:
- [CNN Model](docs/01_CNN_Model.md)
- [MLP Model](docs/02_MLP_Model.md)
- [GraphSAGE Model](docs/03_GNN_GraphSAGE.md)
- [GCN Model](docs/04_GNN_GCN.md)
- [GAT Model](docs/05_GNN_GAT.md)
- [Post Processing](docs/06_Post_Processing.md)
- [Data Analysis](docs/07_Data_Analysis.md)

## Environment Setup

You can set up the environment using Virtual Environments (venv), Conda, or Docker. 

### ⚠️ Important: DGL Installation
The Deep Graph Library (DGL) installation varies depending on your Operating System, PyTorch version, and whether you are using CUDA (GPU) or CPU.
Please consult the [Official DGL Getting Started Page](https://www.dgl.ai/pages/start.html) for your exact configuration.

**Examples:**
- **CUDA 12.1 (Linux/Windows):** `pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html`
- **CUDA 12.4 (Linux/Windows):** `pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html`
- **CPU-only (Mac/Linux/Windows):** `pip install dgl -f https://data.dgl.ai/wheels/repo.html`

---

### Option 1: Python Virtual Environment (venv)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Install the appropriate DGL version (see note above)
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

### Option 2: Conda
```bash
conda env create -f environment.yml
conda activate sln-gnn
# Conda environment.yml attempts to install DGL via dglteam channel.
# If it fails, install manually with pip inside your conda env.
```

### Option 3: Docker
A `Dockerfile` is provided for containerized execution. It defaults to installing the CPU version of DGL.
```bash
docker build -t sln-gnn-env .
docker run -p 8888:8888 sln-gnn-env
```

## Usage
1. Configure your datasets, seeds, and execution mode in `utils.py` or within `Cell 2` of the notebooks. 
2. **Debug Mode:** By default, the notebooks might be set to `DEBUG = True` to run a small subset (1 seed, 1 dataset, limited epochs) for regression testing. Set `DEBUG = False` to run the full paper experiments.
3. Run the notebooks sequentially or focus on `03_GNN_GraphSAGE.ipynb` to evaluate the primary GNN model. Output confusion matrices and models will be saved in the `results/` folder.