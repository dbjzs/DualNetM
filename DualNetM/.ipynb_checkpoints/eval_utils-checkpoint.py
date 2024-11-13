'''
Functions for computing evaluation metrics.
Some codes used for GRN evaluation are referenced from https://github.com/murali-group/Beeline.
'''

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, average_precision_score, auc
from itertools import product, permutations



def EarlyPrec(trueEdgesDF: pd.DataFrame, predEdgesDF: pd.DataFrame,
              weight_key: str = 'weights_combined', TFEdges: bool = True):
    print("Calculating the EPR(early prediction rate)...")

    # Remove self-loops
    trueEdgesDF = trueEdgesDF.loc[(trueEdgesDF['Gene1'] != trueEdgesDF['Gene2'])]
    if 'Score' in trueEdgesDF.columns:
        trueEdgesDF = trueEdgesDF.sort_values('Score', ascending=False)
    trueEdgesDF = trueEdgesDF.drop_duplicates(keep='first', inplace=False).copy()
    trueEdgesDF.reset_index(drop=True, inplace=True)

    predEdgesDF = predEdgesDF.loc[(predEdgesDF['Gene1'] != predEdgesDF['Gene2'])]
    if weight_key in predEdgesDF.columns:
        predEdgesDF = predEdgesDF.sort_values(weight_key, ascending=False)
    predEdgesDF = predEdgesDF.drop_duplicates(keep='first', inplace=False).copy()
    predEdgesDF.reset_index(drop=True, inplace=True)

    uniqueNodes = np.unique(trueEdgesDF.loc[:, ['Gene1', 'Gene2']])
    if TFEdges:
        # Consider only edges going out of source genes
        print("  Consider only edges going out of source genes")

        # Get a list of all possible TF to gene interactions
        possibleEdges_TF = set(product(set(trueEdgesDF.Gene1), set(uniqueNodes)))

        # Get a list of all possible interactions
        possibleEdges_noSelf = set(permutations(uniqueNodes, r=2))

        # Find intersection of above lists to ignore self edges
        # TODO: is there a better way of doing this?
        possibleEdges = possibleEdges_TF.intersection(possibleEdges_noSelf)
        possibleEdges = pd.DataFrame(possibleEdges, columns=['Gene1', 'Gene2'], dtype=str)

        # possibleEdgesDict = {'|'.join(p): 0 for p in possibleEdges}
        possibleEdgesDict = possibleEdges['Gene1'] + "|" + possibleEdges['Gene2']

        trueEdges = trueEdgesDF['Gene1'].astype(str) + "|" + trueEdgesDF['Gene2'].astype(str)
        trueEdges = trueEdges[trueEdges.isin(possibleEdgesDict)]
        print("  {} TF Edges in ground-truth".format(len(trueEdges)))
        numEdges = len(trueEdges)

        predEdgesDF['Edges'] = predEdgesDF['Gene1'].astype(str) + "|" + predEdgesDF['Gene2'].astype(str)
        # limit the predicted edges to the genes that are in the ground truth
        predEdgesDF = predEdgesDF[predEdgesDF['Edges'].isin(possibleEdgesDict)]
        print("  {} Predicted TF edges are considered".format(len(predEdgesDF)))

        M = len(set(trueEdgesDF.Gene1)) * (len(uniqueNodes) - 1)

    else:
        trueEdges = trueEdgesDF['Gene1'].astype(str) + "|" + trueEdgesDF['Gene2'].astype(str)
        trueEdges = set(trueEdges.values)
        numEdges = len(trueEdges)
        print("  {} edges in ground-truth".format(len(trueEdges)))

        M = len(uniqueNodes) * (len(uniqueNodes) - 1)

    if not predEdgesDF.shape[0] == 0:
        # Use num True edges or the number of
        # edges in the dataframe, which ever is lower
        maxk = min(predEdgesDF.shape[0], numEdges)
        edgeWeightTopk = predEdgesDF.iloc[maxk - 1][weight_key]

        nonZeroMin = np.nanmin(predEdgesDF[weight_key].replace(0, np.nan).values)
        bestVal = max(nonZeroMin, edgeWeightTopk)

        newDF = predEdgesDF.loc[(predEdgesDF[weight_key] >= bestVal)]
        predEdges = set(newDF['Gene1'].astype(str) + "|" + newDF['Gene2'].astype(str))
        print("  {} Top-k edges selected".format(len(predEdges)))
    else:
        predEdges = set([])

    if len(predEdges) != 0:
        intersectionSet = predEdges.intersection(trueEdges)
        print("  {} true-positive edges".format(len(intersectionSet)))
        Eprec = len(intersectionSet) / len(predEdges)
        Erec = len(intersectionSet) / len(trueEdges)
    else:
        Eprec = 0
        Erec = 0

    random_EP = len(trueEdges) / M
    EPR = Erec / random_EP
    return EPR


def evaluateAUPRratio(output, label, TFs, Genes):
    label_set_aupr = set(label['Gene1'] + label['Gene2'])
    preds, labels, randoms = [], [], []
    res_d = {}
    l = []
    p = []
    for item in (output.to_dict('records')):
        res_d[item['Gene1'] + item['Gene2']] = item['weights_combined']
    for item in (set(label['Gene1'])):
        for item2 in set(label['Gene1']) | set(label['Gene2']):
            if item + item2 in label_set_aupr:
                l.append(1)
            else:
                l.append(0)
            if item + item2 in res_d:
                p.append(res_d[item + item2])
            else:
                p.append(-1)
    AUPRC=average_precision_score(l, p)
    AUPRC_ratio=average_precision_score(l, p) / np.mean(l)

    return AUPRC, AUPRC_ratio




def evaluateAUROC(output, label):
    score = output.loc[:, ['weights_combined']].values
    label_dict = {}
    for row_index, row in label.iterrows():
        label_dict[row[0] + row[1]] = 1
    test_labels = []
    for row_index, row in output.iterrows():
        tmp = row[0] + str(row[1])
        if tmp in label_dict:
            test_labels.append(1)
        else:
            test_labels.append(0)
    test_labels = np.array(test_labels, dtype=bool).reshape([-1, 1])
    fpr, tpr, threshold = roc_curve(test_labels, score)
    AUROC = auc(fpr, tpr)
    return AUROC
