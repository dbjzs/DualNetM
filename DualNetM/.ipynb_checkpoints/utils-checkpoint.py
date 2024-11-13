import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
import networkx as nx
import scanpy as sc
from typing import Optional, Union
from pathlib import Path
import re

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
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        else:
            adata=sc.read_h5ad(input_expData)

    elif isinstance(input_expData,sc.AnnData):
            adata=input_expData
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
        adata.var_names=adata.var_names.str()
        print('The species is',possible_species)
    else:
        possible_species='human'
        adata.var_names = adata.var_names.str()
        print('The species is', possible_species)


    if isinstance(input_expData,str):
        netData=pd.read_csv(input_priorNet,index_col=None,header=0)
    elif isinstance(input_priorNet,pd.DataFrame):
        netData=input_priorNet.copy()
    else:
        raise Exception("invalid input",input_priorNet)



    netData['from']=netData['from'].str()
    netData['to']=netData['to'].str()
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



    adata=adata[:,priori_network_nodes]
    if lineages is None:
        cells_in_lineage_dict={'all':adata.obs_names}
    else:
        cells_in_lineage_dict = {}
        for l in lineages:
            non_na_cells = adata.obs_names[adata.obs[l].notna()]
            cells_in_lineage_dict[l] = non_na_cells

    print(f"Consider the input data with {len(cells_in_lineage_dict)}lineages:")


    adata_lineages=dict()
    for l,c in cells_in_lineage_dict.items():
        print(f"Lineage-{l}:")
        adata_l=sc.AnnData(X=adata[c,:].to_df())
        adata_l.varm['centrality_prior_net']=centrality
        adata_l.varm['idx_GeneName_map']=idx_GeneName_map
        adata_l.uns['name']=l



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
        adata_lineages[l]=adata_l
        print(f" n_genes x n_cells ={adata_l.n_vars} x {adata_l.n_obs}")

    return adata_lineages



def plot_controllability_metrics(cefcon_results: Union[dict, list], return_value: bool = False):
    """

    """
    #
    con_df = pd.DataFrame(columns=['MDS_controllability_score', 'MFVS_controllability_score',
                                   'Jaccard_index', 'Driver_regulators_coverage', 'Lineage'])
    for k in cefcon_results:
        if isinstance(k, str):
            result = cefcon_results[k]
        else:
            result = k
        drivers_df = result.driver_regulator

        MFVS_driver_set = set(drivers_df.loc[drivers_df['is_MFVS_driver']].index)
        MDS_driver_set = set(drivers_df.loc[drivers_df['is_MDS_driver']].index)
        driver_regulators = set(drivers_df.loc[drivers_df['is_driver_regulator']].index)
        top_ranked_genes = driver_regulators.union(
            (set(drivers_df.index) - MFVS_driver_set.union(MDS_driver_set)))
        N_genes = result.n_genes

        # MDS controllability score
        MDS_con = 1 - len(MDS_driver_set) / N_genes
        # MFVS controllability score
        MFVS_con = 1 - len(MFVS_driver_set) / N_genes
        # Jaccard index
        Jaccard_con = len(MDS_driver_set.intersection(MFVS_driver_set)) / len(MDS_driver_set.union(MFVS_driver_set))
        # driver regulators coverage
        Critical_con = len(driver_regulators) / len(MDS_driver_set.union(MFVS_driver_set))

        con_df.loc[len(con_df)] = {'MDS_controllability_score': MDS_con,
                                   'MFVS_controllability_score': MFVS_con,
                                   'Jaccard_index': Jaccard_con,
                                   'Driver_regulators_coverage': Critical_con,
                                   'Lineage': result.name}

    con_df = pd.melt(con_df, id_vars=['Lineage'])

    # plot
    fig = plt.figure(figsize=(4, 0.5))
    sns.set_theme(style="ticks", font_scale=1.0)
    surrent_palette = sns.color_palette("Set1")

    # Controallability score
    con_df1 = con_df.loc[con_df['variable'].isin(['MDS_controllability_score', 'MFVS_controllability_score'])]
    fig.add_subplot(1, 3, 1)
    ax1 = sns.barplot(x="variable", y="value", hue="Lineage", palette=surrent_palette,
                      data=con_df1)
    ax1.set_xlabel('')
    ax1.set_xticklabels(['MDS', 'MFVS'], rotation=45, ha="right")
    ax1.set_ylabel('Controllability Score')
    ax1.set_ylim(con_df1['value'].min().round(1)-0.1, 1.0)
    sns.despine()
    ax1.get_legend().remove()

    # Jaccard index
    con_df2 = con_df.loc[con_df['variable'].isin(['Jaccard_index'])]
    fig.add_subplot(1, 3, 2)
    ax2 = sns.barplot(x="variable", y="value", hue="Lineage", palette=surrent_palette,
                      data=con_df2)
    ax2.set_xlabel('')
    ax2.set_xticklabels('')
    ax2.set_ylabel(r'Jaccard Index\nbetween MFVS & MDS')
    sns.despine()
    ax2.get_legend().remove()

    # Driver regulators coverage
    con_df3 = con_df.loc[con_df['variable'].isin(['Driver_regulators_coverage'])]
    fig.add_subplot(1, 3, 3)
    ax3 = sns.barplot(x="variable", y="value", hue="Lineage", palette=surrent_palette,
                      data= con_df3)
    ax3.set_xlabel ( '' )
    return con_df








