from multiprocessing import Pool, cpu_count
from scipy.stats import spearmanr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
from typing import Optional, Union
import os

def Co_express_network(gene_express, prior_marker, corr_threshold=0.6, p_val_threshold=0.05, target_size=350):
    """
    Compute coexpression networks for each cell type based on specified genes.
    :param gene_express: DataFrame  gene expression data
    :param prior_marker: DataFrame prior marker genes for different cell types
    :param corr_threshold: Initial correlation threshold
    :param p_val_threshold: p-value threshold
    :param target_size: Target number of significant gene pairs
    :return: Dictionary where each key is a cell type and each value is a tuple containing the correlation, p-value matrices, specified genes, and long format DataFrame
    """
    # Compute full correlation and p-value matrices
    correlation, p_value = spearmanr(gene_express, axis=0)
    correlation_matrix = pd.DataFrame(correlation, index=gene_express.columns, columns=gene_express.columns)
    p_value_matrix = pd.DataFrame(p_value, index=gene_express.columns, columns=gene_express.columns)


    cell_type_indices = [i for i in range(len(prior_marker.columns))]
    coexpression_networks = {}

    for idx in cell_type_indices:
        cell_type = prior_marker.columns[idx]
        specified_genes = prior_marker.iloc[:, idx]
        specified_genes = [gene for gene in specified_genes if gene in correlation_matrix.columns]

        # Filter correlation and p-value matrices for the specified genes
        filtered_correlation_matrix = correlation_matrix.loc[specified_genes, specified_genes]
        filtered_p_value_matrix = p_value_matrix.loc[specified_genes, specified_genes]

        # Generate long format DataFrame
        results = []
        while True:
            for gene in specified_genes:
                if gene not in filtered_correlation_matrix.columns:
                    continue

                # Extract correlations and p-values for the given gene
                correlations = filtered_correlation_matrix[gene]
                p_values = filtered_p_value_matrix[gene]

                # Filter genes based on correlation and p-value thresholds
                filtered_genes = correlations[
                    (correlations > corr_threshold) & (correlations < 0.95) & (p_values < p_val_threshold)
                    ]

                # Save results
                for other_gene, corr_value in filtered_genes.items():
                    results.append({
                        'Candidate_marker': other_gene,
                        'Prior_marker': gene,
                        'correlation': corr_value,
                        'p_value': p_values[other_gene]
                    })

            # Convert results to DataFrame
            long_format = pd.DataFrame(results).drop_duplicates()
            if len(long_format) >= target_size:
                break
            corr_threshold -= 0.005
            if corr_threshold < 0.2:
                break

        # Store in dictionary
        coexpression_networks[cell_type] = (specified_genes, long_format)
    return coexpression_networks




def Co_regulatory_network(coexpression_networks, GRN,output_file: Optional[str] = None):
    def process_group(group, long_format):
        filtered_rows = []
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    gene1_1 = group.iloc[i]['Gene1']
                    gene1_2 = group.iloc[j]['Gene1']
                    if ((long_format['gene1'] == gene1_1) & (long_format['gene2'] == gene1_2)).any() or \
                            ((long_format['gene1'] == gene1_2) & (long_format['gene2'] == gene1_1)).any():
                        filtered_rows.append(group.iloc[j])
                        filtered_rows.append(group.iloc[i])
                        break
        return filtered_rows
    def parallel_filtering(predEdgesDF3, long_format):
        grouped = predEdgesDF3.groupby('Gene2')
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(process_group, [(group, long_format) for _, group in grouped])

        filtered_rows = [item for sublist in results for item in sublist]
        filtered_df2 = pd.DataFrame(filtered_rows)

        return filtered_df2

    def process_group2(group, long_format):
        filtered_rows = []
        if len(group) > 1:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    gene1_1 = group.iloc[i]['Gene2']
                    gene1_2 = group.iloc[j]['Gene2']
                    if ((long_format['gene1'] == gene1_1) & (long_format['gene2'] == gene1_2)).any() or \
                            ((long_format['gene1'] == gene1_2) & (long_format['gene2'] == gene1_1)).any():
                        filtered_rows.append(group.iloc[j])
                        filtered_rows.append(group.iloc[i])
                        break
        return filtered_rows

    def parallel_filtering2(predEdgesDF6, long_format):
        grouped = predEdgesDF6.groupby('Gene1')

        # 使用 Pool 来并行化处理每个 group
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(process_group2, [(group, long_format) for _, group in grouped])

        # 合并所有结果
        filtered_rows = [item for sublist in results for item in sublist]

        filtered_df3 = pd.DataFrame(filtered_rows)

        return filtered_df3
    

    result_dict = {}
    for cell_type in coexpression_networks.keys():
        specified_genes_df = coexpression_networks[cell_type][0]
        long_format_df = coexpression_networks[cell_type][1]

        # Filter GRN based on specified genes and coexpression network
        filtered_df = GRN[GRN['Gene1'].isin(specified_genes_df['gene'])]
        predEdgesDF2 = filtered_df[filtered_df['Gene2'].isin(filtered_df['Gene2'])]
        predEdgesDF3 = predEdgesDF2[predEdgesDF2['Gene1'].isin(long_format_df['gene1'])]
        predEdgesDF3 = predEdgesDF3.drop_duplicates()

        filtered_df2 = parallel_filtering(predEdgesDF3, long_format_df)
        filtered_df2 = filtered_df2.drop_duplicates()




        grouped = filtered_df2.groupby('Gene2')
        similarity_results = []
        for gene2, group in grouped:
            marker_pairs = group[group['Gene1'].isin(specified_genes_df['gene'])][['Gene1', 'Gene2', 'weights_combined']]
            all_pairs = group[['Gene1', 'Gene2', 'weights_combined']]
            for idx1, pair1 in marker_pairs.iterrows():
                for idx2, pair2 in all_pairs.iterrows():
                    if idx1 != idx2:
                        weight1 = pair1['weights_combined']
                        weight2 = pair2['weights_combined']
                        abs_diff = abs(weight1 - weight2)
                        relative_diff = abs_diff / max(weight1, weight2)

                        similarity_results.append({
                            'Prior_marker': pair1['Gene1'],
                            'Target_gene': pair1['Gene2'],
                            'Candidate_marker': pair2['Gene1'],
                            'Target_gene': pair2['Gene2'],
                            'relative_difference': relative_diff
                        })
        similarity_df = pd.DataFrame(similarity_results)
        similarity_df = similarity_df.drop_duplicates()
        sorted_df = similarity_df.sort_values(by='relative_difference')
        top_50_percent_df = sorted_df.iloc[:int(len(sorted_df) * 0.5)]

        Candidate_marker_out_number = top_50_percent_df['Candidate_marker'].value_counts()

        filtered_df2 = GRN[GRN['Gene2'].isin(specified_genes_df['gene'])]
        predEdgesDF5 = GRN[GRN['Gene1'].isin(filtered_df2['Gene1'])]
        predEdgesDF6 = predEdgesDF5[predEdgesDF5['Gene2'].isin(long_format_df['gene1'])]
        predEdgesDF6 = predEdgesDF6.drop_duplicates()
        filtered_df3 = parallel_filtering2(predEdgesDF6, long_format_df)
        filtered_df3=filtered_df3.drop_duplicates()


        grouped = filtered_df3.groupby('Gene1')
        similarity_results = []
        for gene2, group in grouped:
            # 提取当前分组中的 marker 基因对和所有基因对
            marker_pairs = group[group['Gene2'].isin(specified_genes_df['gene'])][['Gene1', 'Gene2', 'weights_combined']]
            all_pairs = group[['Gene1', 'Gene2', 'weights_combined']]

            # 计算 marker 基因对的权重与所有基因对的权重之间的相对差值
            for idx1, pair1 in marker_pairs.iterrows():
                for idx2, pair2 in all_pairs.iterrows():
                    if idx1 != idx2:  # 避免计算自己与自己的差值
                        weight1 = pair1['weights_combined']
                        weight2 = pair2['weights_combined']
                        abs_diff = abs(weight1 - weight2)
                        relative_diff = abs_diff / max(weight1, weight2)

                        similarity_results.append({
                            'TFs': pair1['Gene1'],
                            'Prior_marker': pair1['Gene2'],
                            'TFs': pair2['Gene1'],
                            'Candidate_other': pair2['Gene2'],
                            'relative_difference': relative_diff
                        })

        similarity_df = pd.DataFrame(similarity_results)
        similarity_df = similarity_df.drop_duplicates()
        sorted_df = similarity_df.sort_values(by='relative_difference')
        top_50_percent_df = sorted_df.iloc[:int(len(sorted_df) * 0.5)]
        Candidate_marker_in_number = top_50_percent_df['Candidate_other'].value_counts()

        Candidate_marker_in_number.columns =['Gene','number_in']
        Candidate_marker_out_number.columns =['Gene','number_out']
        combined_df = pd.merge(Candidate_marker_in_number, Candidate_marker_out_number, on='Gene', how='outer')
        combined_df['number_in'].fillna(0, inplace=True)
        combined_df['number_out'].fillna(0, inplace=True)
        combined_df['Total_number'] = combined_df['number_in'] + combined_df['number_out']
        combined_df_sorted = combined_df.sort_values(by='Total_number', ascending=False)

        result_dict[cell_type] = combined_df_sorted
        #output
        if output_file is not None:
            cell_output_file = os.path.join(output_file, f"{cell_type}_Candidate_marker.csv")
            combined_df_sorted.to_csv(cell_output_file)

    return result_dict


