# DualNetM-new
An Adaptive Attention-Driven Dual Network Framework for Inferring Functional Markers

DualNetM is a computational tool for Inferring Functional Markers from single-cell RNA-seq data.
It takes a prior gene interaction network,expression profiles and prior markers from scRNA-seq data as inputs, and consists of three main components, including gene 
regulatory network (GRN) construction, Functional Markers inference and Markers activity score.

![workframe.svg](/workframe.svg)

## Installation
DualNetM was tested with Python (3.10.13). 
We recommend running DualNetM on CUDA. 
The following packages are required to be able to run this code:

### Requirements
- [python(3.10.13)](https://www.python.org/)
- [pytorch(2.2.0)](https://pytorch.org/get-started/locally/) 
- [torch-geometric(>=2.1.0)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [scanpy(1.9.8)](https://scanpy.readthedocs.io/en/stable/installation.html)
- networkx(3.2.1)
- cvxpy(1.4.2)
- [pyscenic(0.12.1)](https://pyscenic.readthedocs.io/en/latest/installation.html)
- numpy, scipy, pandas, scikit-learn, tqdm
- Recommended: An NVIDIA GPU with CUDA support for GPU acceleration
