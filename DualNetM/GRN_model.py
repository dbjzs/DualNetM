from typing import Optional, Union
from torch_geometric.typing import Adj
import concurrent.futures
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import trange
import math
from scanpy import AnnData
import os
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
import torch_geometric as pyg
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import Linear, BatchNorm, DeepGraphInfomax
from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    softmax,
    to_undirected,
)

from torch_geometric.data import Data
from tqdm import tqdm
from DualNetM.DualNetM_result import DualNetMResult
import csv

class GraphAttention_layer(MessagePassing):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 attention_type: str = 'Gaussian',
                 flow: str = 'source_to_target',
                 heads: int = 1,
                 concat: bool = True,
                 dropout: float = 0.0,
                 add_self_loops: bool = False,
                 to_undirected: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', flow)
        super(GraphAttention_layer, self).__init__(node_dim=0, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.attention_type = attention_type
        self.to_undirected = to_undirected


        self.lin_l = Linear(input_dim, heads * output_dim, bias=False,
                            weight_initializer='glorot')
        self.lin_r = Linear(input_dim, heads * output_dim, bias=False,
                            weight_initializer='glorot')

        self.register_parameter('att_l', None)
        self.register_parameter('att_r', None)

        if concat:
            self.bias = nn.Parameter(Tensor(heads * output_dim))
            self.weight_concat = nn.Parameter(Tensor(heads * output_dim, output_dim))

        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)
        glorot(self.weight_concat)

    def forward(self, x: Tensor, edge_index: Adj, return_attention_weights: Optional[bool] = None):
        N, H, C = x.size(0), self.heads, self.output_dim
        if self.to_undirected:
            edge_index = to_undirected(edge_index)
        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        else:
            edge_index, _ = remove_self_loops(edge_index)

        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)

        out = self.propagate(edge_index, x=(x_l, x_r), size=None)
        alpha = self._alpha
        self._alpha = None


        if self.concat:
            out = out.view(-1, self.heads * self.output_dim) #out:[num_edges, heads, output_dim]-->[num_edges, heads*output_dim]
            out += self.bias
            out = torch.matmul(out, self.weight_concat) #[num_edges, output_dim]

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, (edge_index, alpha)



    def message(self, edge_index_i: Tensor, x_i: Tensor, x_j: Tensor, size_i: Optional[int]):
        Tau = 0.25
        if self.attention_type == 'Gaussian':
            distances = torch.norm(x_i - x_j, p=2, dim=-1)
            sigma = distances.std().item()
            alpha = torch.exp(-distances ** 2 / (2 * sigma ** 2))
            alpha=1-alpha
        alpha = softmax(alpha/Tau, edge_index_i, num_nodes=size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, {}, heads={}, type={})'.format(self.__class__.__name__,
                                                      self.input_dim,
                                                      self.output_dim,
                                                      self.heads,
                                                      self.attention_type)




class GRN_Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 heads_num: int = 4,
                 dropout: float = 0.0,
                 attention_type: str = 'Gaussian'):
        super(GRN_Encoder, self).__init__()

        self.att_weights_first = None
        self.att_weights_second = None
        self.att_weights_third=None
        self.x_embs = None

        self.x_input = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.layers = nn.ModuleList([])
        dims = [hidden_dim,hidden_dim,hidden_dim]
        for l in range(len(dims)):
            concat = True  # if l==0 else False
            last_dim = hidden_dim if l < len(dims) - 1 else output_dim
            self.layers.append(nn.ModuleList([
                BatchNorm(dims[l]),
                GraphAttention_layer(dims[l], dims[l], heads=heads_num,
                                     concat=concat, dropout=dropout,
                                     attention_type=attention_type,
                                    ),
                nn.Sequential(
                    nn.Linear(dims[l], hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, last_dim),
                ),
            ]))
        self.project = nn.Linear(output_dim, output_dim * 4)



    def forward(self, data: dict):
        x, edge_index = data['x'], data['edge_index']
        x = self.x_input(x) #Linear


        att_weights_in = []
        for norm, attn_in,ffn in self.layers:
            x = norm(x)
            x_in, att_weights_in_ = attn_in(x, edge_index, return_attention_weights=True)
            x = ffn(self.act(x_in))
            att_weights_in.append(att_weights_in_)


        self.x_embs = x

        self.att_weights_first = (att_weights_in[0][0], att_weights_in[0][1])
        self.att_weights_second = (att_weights_in[1][0], att_weights_in[1][1])
        self.att_weights_third=(att_weights_in[2][0], att_weights_in[2][1])

        return self.project(x)



#
class NetModel(object):
    """Summary of class here.

    description

    Attributes:
        hidden_dim:
        output_dim:
        heads_first:

    """

    def __init__(self,
                 hidden_dim: int = 128,
                 output_dim: int = 64,
                 heads: int =4,
                 attention_type: str = 'Gaussian',
                 dropout: float = 0.1,
                 epochs: int = 340,
                 repeats: int = 1,
                 seed: int = -1,
                 cuda: int = -1,
                 weight_decay=4e-3,
                 lr=1e-3,
                 ):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.heads = heads
        self.lr=lr
        self.weight_decay=weight_decay
        self.attention_type = attention_type
        self.dropout = dropout
        self.epochs = epochs
        self.repeats = repeats
        if seed > -1:
            pyg.seed_everything(seed)
            torch.backends.cudnn.deterministic = True
        self.cuda = cuda

        self._idx_GeneName_map = None
        self._att_coefs = None
        self._node_embs = None
        self._adata = None
        self.GRN_predicted = None


    def __get_PYG_data(self, adata: AnnData) -> Data:
        # edge index
        source_nodes = adata.uns['edgelist']['from'].tolist()
        target_nodes = adata.uns['edgelist']['to'].tolist()
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

        # Get pytorch_data
        x = torch.from_numpy(adata.to_df().T.to_numpy())
        pyg_data = Data(x=x, edge_index=edge_index)

        self._idx_GeneName_map = adata.varm['idx_GeneName_map']
        self._adata = adata
        return pyg_data

    @staticmethod
    def __corruption(data: Data) -> Data:
        x, edge_index = data['x'], data['edge_index']
        data_neg = Data(x=x[torch.randperm(x.size(0))], edge_index=edge_index)
        return data_neg

    @staticmethod
    def __summary(z, *args, **kwargs) -> torch.Tensor:
        # return torch.sigmoid(z.mean(dim=0))
        return torch.sigmoid(torch.cat((3 * z.mean(dim=0).unsqueeze(0),
                                        z.max(dim=0)[0].unsqueeze(0),
                                        z.min(dim=0)[0].unsqueeze(0),
                                        2 * z.median(dim=0)[0].unsqueeze(0),
                                        ), dim=0))

    @staticmethod
    def __train(data, model, optimizer):
        model.train()
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(data)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    @staticmethod
    def __get_encoder_results(data, model):
        model.eval()
        emb_last = model(data)
        return model.x_embs, model.att_weights_first, model.att_weights_second,model.att_weights_third, emb_last



    def run(self, adata: AnnData, showProgressBar: bool = True):
        if self.cuda == -1:
            device = "cpu"
        else:
            device = 'cuda:%s' % self.cuda


        data = self.__get_PYG_data(adata).to(device)#Data for pyg input
        input_dim = data.num_node_features

        ## Run for many times and take the average
        att_weights_all = []
        emb_out_avg = 0
        for rep in range(self.repeats):
            ## Encoder & Model & Optimizer
            encoder = GRN_Encoder(input_dim, self.hidden_dim, self.output_dim, self.heads,
                                  dropout=self.dropout, attention_type=self.attention_type).to(device)
            DGI_model = DeepGraphInfomax(hidden_channels=self.output_dim * 4,
                                         encoder=encoder,
                                         summary=self.__summary,
                                         corruption=self.__corruption).to(device)
            optimizer = torch.optim.Adam(DGI_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

                ## Train
            best_encoder = encoder.state_dict()
            min_loss = np.inf
            if showProgressBar:
                with trange(self.epochs, ncols=100) as t:
                    for epoch in t:
                        loss = self.__train(data, DGI_model, optimizer)
                        t.set_description('  Iter: {}/{}'.format(rep + 1, self.repeats))
                        if epoch < self.epochs - 1:
                            t.set_postfix(loss=loss)

                        if min_loss > loss:
                            min_loss = loss
                            best_encoder = encoder.state_dict()

                        if epoch == self.epochs - 1:
                            t.set_postfix(loss=loss, min_loss=min_loss)
            else:
                print('  Iter: {}/{}'.format(rep + 1, self.repeats), end='... ')
                for epoch in range(self.epochs):
                    loss = self.__train(data, DGI_model, optimizer)
                    if min_loss > loss:
                        min_loss = loss
                        best_encoder = encoder.state_dict()
                print('Min_train_loss: {}'.format(min_loss))


                ## Get the result of the best model
                # encoder = GRN_Encoder(input_dim, self.hidden_dim, self.output_dim, self.heads_first,
                #                       dropout=self.dropout, attention_type=self.attention_type).to(device)
            encoder.load_state_dict(best_encoder)
            gene_emb, weights_first, weights_second, weights_third,emb_last = self.__get_encoder_results(data, encoder)
            gene_emb = gene_emb.cpu().detach().numpy()

            weights_first = (weights_first[0].cpu().detach(), weights_first[1].mean(dim=1, keepdim=True).cpu().detach())
            weights_second = (weights_second[0].cpu().detach(), weights_second[1].mean(dim=1, keepdim=True).cpu().detach())
            weights_third = (weights_third[0].cpu().detach(), weights_third[1].mean(dim=1, keepdim=True).cpu().detach())
    
            att_weights = (weights_first[1] + weights_second[1]+weights_third[1])/3
            att_weights_all.append(att_weights)

            emb_out_avg += gene_emb
            if device == 'cuda':
                torch.cuda.empty_cache()

        if self.repeats > 1:
            att_weights_all = torch.stack((att_weights_all), 0)
        else:
            att_weights_all = att_weights_all[0].unsqueeze(0)
        emb_out_avg = emb_out_avg / self.repeats

        self.edge_index = data.edge_index.cpu()
        self._att_coefs = (weights_first[0], att_weights_all)
        self._node_embs = emb_out_avg


    def get_network(self,output_file: Optional[str] = None) -> nx.DiGraph:

        edge_index_ori = self.edge_index
        edge_index_with_selfloop, att_coefs_with_selfloop = self._att_coefs[0], self._att_coefs[1]
        #att_coefs_with_selfloop:[num_repeats, num_edges, 2]

        ori_att_coefs_all = pd.DataFrame(
            {'from': edge_index_with_selfloop[0].numpy().astype(int),
             'to': edge_index_with_selfloop[1].numpy().astype(int),
             'att_coef': att_coefs_with_selfloop.mean(0, keepdim=False)[:, 0].numpy()}
        )

        ori_att_coefs_all['edge_idx_tmp'] = ori_att_coefs_all['from'].astype(str) + "|" \
                                            + ori_att_coefs_all['to'].astype(str)


        # Scale the weights
        scaled_att_coefs = []
        g = nx.from_edgelist(edge_index_with_selfloop.numpy().T, create_using=nx.DiGraph)

        att_coef_i = att_coefs_with_selfloop[:, :, 0]
        d_out = pd.DataFrame(g.out_degree(), columns=['index', 'degree'])
        d_out.index = d_out['index']
        att_coef_i = att_coef_i * np.array(d_out.loc[edge_index_with_selfloop[0, :].numpy(), 'degree'])
        att_coef_i = att_coef_i.T

        #remove_self_loops
        edge_index, att_coef_i = remove_self_loops(edge_index_with_selfloop, att_coef_i)
        scaled_att_coefs = scaled_att_coefs + [att_coef_i.clone()]

        scaled_att_coefs_all = pd.DataFrame(
            {'from': edge_index[0].numpy().astype(int),
             'to': edge_index[1].numpy().astype(int),
             'weights': scaled_att_coefs[0].mean(1, keepdim=False).numpy()}
        )


        att_weights_combined = scaled_att_coefs_all['weights']
          # All edges

        filtered_edge_idx = list(range(len(att_weights_combined)))
        scaled_att_coefs_filtered = scaled_att_coefs_all.iloc[filtered_edge_idx, :].copy()

        ## Output the scaled attention coefficient of the predicted network
        ori_att_coefs_filtered = ori_att_coefs_all.loc[
            ori_att_coefs_all['edge_idx_tmp'].isin(
                scaled_att_coefs_filtered['from'].astype(str) + "|" + scaled_att_coefs_filtered['to'].astype(str)),
            ['from', 'to', 'att_coef']
        ].copy()

        net_filtered_df = pd.merge(scaled_att_coefs_filtered, ori_att_coefs_filtered, on=['from', 'to'], how='inner')


        ## To networkx
        G_nx = nx.from_pandas_edgelist(net_filtered_df, source='from', target='to', edge_attr=True,
                                       create_using=nx.DiGraph)

        ## Use gene name as index
        mappings = self._idx_GeneName_map.loc[self._idx_GeneName_map['idx'].isin(G_nx.nodes()), ['idx', 'geneName']]
        mappings = {idx: geneName for (idx, geneName) in np.array(mappings)}
        G_nx = nx.relabel_nodes(G_nx, mappings)
        self.GRN_predicted = G_nx

        ## Save the predicted network to file
        if isinstance(output_file, str):
            output_path = os.path.join(output_file, "outputGRN.csv")
            nx.write_edgelist(G_nx, output_path, delimiter=',', data=['weights'])
        return G_nx


    def get_gene_embedding(self, output_file: Optional[str] = None) -> pd.DataFrame:
        if self.GRN_predicted is None:
            raise ValueError(
                f'Did not find the predicted network. Run `NetModel.get_network` first.'
            )
        emb = pd.DataFrame(self._node_embs, index=self._idx_GeneName_map['geneName'])
        emb = emb.loc[emb.index.isin(self.GRN_predicted.nodes), :]
        emb.index = emb.index.astype(str)

        if isinstance(output_file, str):
            emb.to_csv(output_file, index_label='geneName')

        return emb


    def get_DualNetM_results(self, Prior_marker, output_file: Optional[str] = None) -> DualNetMResult:

        network = self.get_network(output_file=output_file)
        results = DualNetMResult(adata=self._adata,
                                GRN=network,
                                Prior_marker=Prior_marker)

        return results
