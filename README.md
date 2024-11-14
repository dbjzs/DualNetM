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
