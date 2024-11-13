import argparse
from os import fspath
from pathlib import Path
import pandas as pd

import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, parent_dir)

from DualNetM.GRN_model import NetModel
from DualNetM.DualNetM_result import DualNetMResult
from DualNetM.utils import data_preparation



def main():
    parser = argparse.ArgumentParser(prog='DualNetM', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_main_args(parser)
    args = parser.parse_args()

    ## output dir
    p = Path(args.out_dir)
    if not p.exists():
        Path.mkdir(p)

    ## load data
    data = data_preparation(args.input_expData, args.input_priorNet)
    prior_marker=pd.read_csv(args.input_prior_markerData)

    ## GRN construction
    DualNetM_GRN = NetModel(hidden_dim=args.hidden_dim,
                                output_dim=args.output_dim,
                                heads=args.heads,
                                epochs=args.epochs,
                                repeats=args.repeats,
                                seed=args.seed,
                                cuda=args.cuda,
                                )
    DualNetM_GRN.run(data, showProgressBar=True)
    G_predicted = DualNetM_GRN.get_network()    
    DualNetM_results = DualNetMResult(adata=DualNetM_GRN._adata,
                                   GRN=G_predicted,
                                   Prior_marker=prior_marker)

    ## find_marker
    DualNetM_results.find_marker(args.out_dir)
    DualNetM_results.plo



def add_main_args(parser: argparse.ArgumentParser):
    # Input data
    input_parser = parser.add_argument_group(title='Input data options')
    input_parser.add_argument('--input_expData', type=str, required=True, metavar='PATH',
                              help='path to the input gene expression data')
    input_parser.add_argument('--input_priorNet', type=str, required=True, metavar='PATH',
                              help='path to the input prior gene interaction network')
    input_parser.add_argument('--input_prior_markerData', type=str, required=True, metavar='PATH',
                              help='path to the input prior marker')
    
    
    # GRN
    grn_parser = parser.add_argument_group(title='Cell-lineage-specific GRN construction options')
    grn_parser.add_argument('--cuda', type=int, default=0,
                            help="an integer greater than -1 indicates the GPU device number and -1 indicates the CPU device")
    grn_parser.add_argument('--seed', type=int, default=-1,
                            help="random seed (set to -1 means no random seed is assigned)")

    grn_parser.add_argument("--hidden_dim", type=int, default=128,
                            help="hidden dimension of the GNN encoder")
    grn_parser.add_argument("--output_dim", type=int, default=64,
                            help="output dimension of the GNN encoder")
    grn_parser.add_argument("--heads", type=int, default=4,
                            help="number of heads")
    grn_parser.add_argument('--epochs', type=int, default=340,
                            help='number of epochs for one run')
    grn_parser.add_argument('--repeats', type=int, default=1,
                            help='number of run repeats')


    # Output dir
    parser.add_argument("--out_dir", type=str, required=True, default='./output',
                        help="results output path")

    return parser


if __name__ == "__main__":
    main()
