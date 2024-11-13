from typing import Optional
import pandas as pd
import networkx as nx
import scanpy as sc
import matplotlib.ticker as ticker
import seaborn as sns
from DualNetM.marker_model import Co_regulatory_network,Co_express_network
import matplotlib.pyplot as plt
from matplotlib_venn import venn2_unweighted
from scipy import sparse

class DualNetMResult:

    def __init__(self,
                 adata: sc.AnnData,
                 GRN: nx.DiGraph,
                 Prior_marker: pd.DataFrame):
        self.name = adata.uns['name']
        self.GRN = GRN
        self.Prior_marker = Prior_marker
        self.result=None
        self.coexpression_networks = None
        genes = list(GRN.nodes())
        #self.expression_data = adata[:, genes].to_df()
        if 'raw_count' in adata.layers:
            raw_data = adata.layers['raw_count']
            self.expression_data = pd.DataFrame(
                raw_data[:, adata.var_names.isin(genes)].toarray() if sparse.issparse(raw_data) else raw_data[:, adata.var_names.isin(genes)],
                index=adata.obs_names,
                columns=genes
            )
        else:
            print("No 'raw_count' found in adata.layers")
        self.n_cells = adata.n_obs
        self.n_genes = len(genes)
        self.n_edges = GRN.number_of_edges()
        self._adata_gene = None
    def __repr__(self) -> str:
        descr = f"DualNetM GRN have n_cells * n_genes = {self.n_cells} * {self.n_genes}, n_edges = {self.n_edges}"
        descr += f"\n    name: {self.name}" \
                 f"\n    expression_data: yes" \
                 f"\n    network: {self.network}"
        return descr



    def find_marker(self, output_file: Optional[str] = None):

        self.coexpression_networks=Co_express_network(gene_express=self.expression_data,
                                                      prior_marker=self.Prior_marker)
        
        self.result = Co_regulatory_network(coexpression_networks=self.coexpression_networks,
                                            GRN=self.GRN,
                                            output_file=output_file)
        return self.coexpression_networks,self.result
        
        
        



    def plot_candidate_marker(self,celltype, topK: int = 20):

        if celltype in self.result:
            combined_df_sorted = self.result[celltype]
            
            top_combined_df = combined_df_sorted.head(topK)

            
            Prior_marker = self.coexpression_networks[celltype][0]
            Prior_marker = pd.DataFrame(Prior_marker, columns=['gene'])

            plt.figure(figsize=(15, len(top_combined_df) * 0.25))
            
            sns.set_theme(style='ticks', font_scale=1)
            
            fig, ax = plt.subplots(figsize=(2.5, len(top_combined_df) * 0.3))

            for i, (gene, count_x, count_y) in enumerate(zip(top_combined_df['Gene'], top_combined_df['number_in'], top_combined_df['number_out'])):
                ax.barh(gene, count_x, color='#6495ED', edgecolor='none', height=0.65)
                ax.barh(gene, count_y, left=count_x, color='#FF6A6A', edgecolor='none', height=0.65)
            plt.xticks(fontsize=13, color='black')
            plt.yticks(fontsize=11, color='black')
            for tick in ax.get_yticklabels():
                gene_name = tick.get_text()  # 获取当前标签文本
                if gene_name in Prior_marker['gene'].values:  # 判断当前 Gene 是否在 marker['gene'] 中
                    tick.set_color('black')  # 在 marker 中的 Gene 设置为黑色
                else:
                    tick.set_color('red')  # 其他 Gene 设置为红色
            
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.yaxis.set_minor_locator(ticker.NullLocator())
            sns.despine()

            ax.set_ylabel('')
            
            plt.xlabel('Total_number', fontsize=14, color='black')
            ax.set_ylim(-0.5, len(top_combined_df) - 0.5)
            ax.invert_yaxis()
            ax.legend().remove()
            plt.show()
        else:
            print(f"Cell type '{celltype}' not found in the results.")



    def plot_Prior_Candidate_Venn(self,celltype):
        if celltype in self.result:
            combined_df_sorted = self.result[celltype]
            Prior_marker = self.coexpression_networks[celltype][0]
            Prior_marker = pd.DataFrame(Prior_marker, columns=['gene'])
            set_candidate = set(combined_df_sorted['Gene'])
            set_marker = set(Prior_marker['gene'])

            plt.figure(figsize=(8, 8))
            venn = venn2_unweighted([set_marker, set_candidate],
                                    ('Prior_marker', 'Candidate_Marker'),
                                    set_colors=('grey', 'red',),
                                    normalize_to=5)

            if venn.set_labels is not None:
                for text in venn.set_labels:
                    text.set_text('')

            for text in venn.subset_labels:
                if text:
                    text.set_fontsize(30)
            #plt.savefig('', dpi=500, bbox_inches='tight')
            plt.show()
        else:
            print(f"Cell type '{celltype}' not found in the results.")




