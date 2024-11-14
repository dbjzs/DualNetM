# DualNetM-new
An Adaptive Attention-Driven Dual Network Framework for Constructing gene regulatary networks and Inferring functional markers

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

#### Setup a conda environment
```
conda create -y --name DualNetM python=3.10.13
conda activate DualNetM
```
#### Install using conda
```
conda install --yes --file requirements.txt
```

## Usage 
### Inferring gene regulatary networks and functional markers from scRNA-seq data
#### Command line usage
```
python DualNetM.py --input_expData /home/dbj/cancer/top3000expressT.csv --input_priorNet /home/dbj/DualNetM-main/data/NicheNet_human.csv --input_prior_markerData /home/dbj/cancer/marker333.csv --out_dir /home/dbj/DualNetM-main/cancer
```
• `input_expData`: a '.csv' file in which rows represent cells and columns represent genes, or a '.h5ad' formatted file with AnnData objects, which is recommended to use the top3000 hypervariable genes.  
• `input_priorNet`: a '.csv' file contains collected a priori gene regulatory relationships, the default is the [NicheNet](https://github.com/saeyslab/nichenetr/tree/master/data).  
• `input_prior_markerData`: a '.csv' file contains a priori markers for different cell types corresponding to the expression data, columns represent priori markers of different cell types, and the column index is celltype.  
• `out_dir`: the path of DualNetM output(Gene regulatory networks and functional markers for different cell types).

#### Package usage
**Quick start by an example ([Jupyter Notebook](run.ipynb)).**
```python
import DualNetM as dm

# Data preparation
adata=sc.read_csv('/home/dbj/cancer/top3000expressT.csv')
Prior_marker=pd.read_csv('/home/dbj/cancer/marker333.csv')
output_file='/home/dbj/DualNetM-main/cancer/'
prior_network = dm.datasets.load_human_prior_network()
data = dm.data_preparation(adata, prior_network)

# Construct GRN
DualNetM_GRN_model = dm.NetModel(epochs=340, cuda='0', seed=9)
DualNetM_GRN_model.run(data)

# Get gene regulatory networks
DualNetM_GRN_model.get_network(output_file=output_file)

#Inferring functional markers
DualNetMResult=DualNetM_GRN_model.get_DualNetM_results(Prior_marker=Prior_marker)
DualNetMResult.find_marker(output_file=output_file)
```
