import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
import networkx as nx
import scanpy as sc
from typing import Optional, Union
from pathlib import Path
import re
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr


def data_preparation(input_expData:Union[str,sc.AnnData,pd.DataFrame],
                    input_priorNet:Union[str,pd.DataFrame])->dict[str:AnnData]:
    print('Data loading and preprocessing....')

    lineages=None
    if isinstance(input_expData,str):
        p=Path(input_expData)
        if p.suffix=='.csv':
            adata=sc.read_csv(input_expData,first_column_names=True)
            adata.raw = adata.copy()
            adata.layers['raw_count'] = adata.raw.X.copy()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        else:
            adata=sc.read_h5ad(input_expData)

    elif isinstance(input_expData,sc.AnnData):
            adata=input_expData
            adata.raw = adata.copy()
            adata.layers['raw_count'] = adata.raw.X.copy()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            lineages=adata.uns.get('lineages')
    elif isinstance(input_expData,pd.DataFrame):
            adata=sc.AnnData(X=input_expData)
    else:
        raise Exception("Invalid input! The input format must be '.csv' file or '.h5ad' "
                        "formatted file, or an 'AnnData' object!", input_expData)


    if bool(re.search('[a-z]',adata.var_names[0])):
        possible_species='mouse'
        adata.var_names=adata.var_names
    else:
        possible_species='human'
        adata.var_names = adata.var_names


    if isinstance(input_expData,str):
        netData=pd.read_csv(input_priorNet,index_col=None,header=0)
    elif isinstance(input_priorNet,pd.DataFrame):
        netData=input_priorNet.copy()
    else:
        raise Exception("invalid input",input_priorNet)


    netData=netData.loc[netData['from'].isin(adata.var_names.values) & netData['to'].isin(adata.var_names.values),:]
    netData=netData.drop_duplicates(subset=['from','to'],keep='first',inplace=False)



    priori_network=nx.from_pandas_edgelist(netData,source='from',target='to',create_using=nx.DiGraph)
    priori_network_nodes=np.array(priori_network.nodes())

    #get g.degree
    in_degree=pd.DataFrame.from_dict(nx.in_degree_centrality(priori_network),
                                     orient='index',columns=['in_degree'])
    out_degree=pd.DataFrame.from_dict(nx.out_degree_centrality(priori_network),
                                      orient='index',columns=['out_degree'])
    centrality=pd.concat([in_degree,out_degree],axis=1)
    centrality=centrality.loc[priori_network_nodes,:]




    idx_GeneName_map=pd.DataFrame({'idx':range(len(priori_network_nodes)),
                                   'geneName':priori_network_nodes},
                                  index=priori_network_nodes)
    edgelist=pd.DataFrame({'from':idx_GeneName_map.loc[netData['from'].tolist(),'idx'].tolist(),
                           'to':idx_GeneName_map.loc[netData['to'].tolist(),'idx'].tolist()})


    adata = adata[:,priori_network_nodes]

    adata_l = adata.copy()
    adata_l.varm['centrality_prior_net'] = centrality
    adata_l.varm['idx_GeneName_map'] = idx_GeneName_map
    adata_l.layers['raw_count'] = adata.layers['raw_count']
    



    if isinstance(adata_l.X,sparse.csr_matrix):
        gene_exp=pd.DataFrame(adata_l.X.A.T,index=priori_network_nodes)
    else:
        gene_exp=pd.DataFrame(adata_l.X.T,index=priori_network_nodes)


    ori_edgeNum = len(edgelist)


    SCC,p=spearmanr(gene_exp,axis=1)
    edges_corr=np.absolute(SCC)
    np.fill_diagonal(edges_corr,0.0)
    x,y=np.where(edges_corr > 0)



    addi_top_edges = pd.DataFrame({'from': x, 'to': y, 'weight': edges_corr[x, y]})

    if ori_edgeNum < 80000:
        addi_top_k=int(ori_edgeNum)
    else:
        addi_top_k=int(ori_edgeNum*0.05)      


    addi_top_edges = addi_top_edges.sort_values(by=['weight'], ascending=False)
    addi_top_edges = addi_top_edges.iloc[0:addi_top_k, 0:2]


    edgelist = pd.concat([edgelist, addi_top_edges.iloc[:, 0:2]], ignore_index=True)
    edgelist = edgelist.drop_duplicates(subset=['from', 'to'], keep='first', inplace=False)
    print('    {} extra edges (Spearman correlation > 0) are added into the prior network.\n'
          '    Total number of edges: {}.'.format((len(edgelist) - ori_edgeNum), len(edgelist)))

    adata_l.uns['edgelist'] = edgelist
    print(f" n_genes x n_cells ={adata_l.n_vars} x {adata_l.n_obs}")

    return adata_l












