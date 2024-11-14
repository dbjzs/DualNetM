# DualNetM-new
An Adaptive Attention-Driven Dual Network Framework for gene regulatary networks and functional markers

DualNetM is a computational tool for Inferring Functional Markers from single-cell RNA-seq data.
It takes a prior gene interaction network,expression profiles and prior markers from scRNA-seq data as inputs, and consists of two main components, including gene 
regulatory network (GRN) construction, functional markers inference 

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

## Usage 
### Inferring gene regulatary networks and functional markers from scRNA-seq data
#### Command line usage
```
python DualNetM.py --input_expData /home/dbj/cancer/top3000expressT.csv --input_priorNet /home/dbj/DualNetM-main/data/NicheNet_human.csv --input_prior_markerData /home/dbj/cancer/marker333.csv --out_dir /home/dbj/DualNetM-main/cancer
```
• `input_expData`: a '.csv' file in which rows represent cells and columns represent genes, or a '.h5ad' formatted file with AnnData objects, which is recommended to use the top3000 hypervariable genes.  
• `input_priorNet`: a '.csv' file contains collected a priori gene regulatory relationships, the default is the [NicheNet](https://github.com/saeyslab/nichenetr/tree/master/data).  
• `input_prior_markerData`: a '.csv' file contains a priori markers for different cell types corresponding to the expression data, columns represent priori markers of different cell types, and the column index is celltype.
