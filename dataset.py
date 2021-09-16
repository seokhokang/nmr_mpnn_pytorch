import os, sys
import numpy as np
import torch
from dgl.convert import graph


class GraphDataset():

    def __init__(self, name='nmrshiftdb2'):

        self.name = name
        self.load()


    def load(self):
        [mol_dict] = np.load('./data/dataset_graph.npz', allow_pickle=True)['data']

        self.n_node = mol_dict['n_node']
        self.n_edge = mol_dict['n_edge']
        self.node_attr = mol_dict['node_attr']
        self.edge_attr = mol_dict['edge_attr']
        self.src = mol_dict['src']
        self.dst = mol_dict['dst']
                
        self.shift = mol_dict['shift']
        self.mask = mol_dict['mask']
        self.smi = mol_dict['smi']

        self.n_csum = np.concatenate([[0], np.cumsum(self.n_node)])
        self.e_csum = np.concatenate([[0], np.cumsum(self.n_edge)])
        

    def __getitem__(self, idx):

        g = graph((self.src[self.e_csum[idx]:self.e_csum[idx+1]], self.dst[self.e_csum[idx]:self.e_csum[idx+1]]), num_nodes = self.n_node[idx])
        g.ndata['node_attr'] = torch.from_numpy(self.node_attr[self.n_csum[idx]:self.n_csum[idx+1]]).float()
        g.edata['edge_attr'] = torch.from_numpy(self.edge_attr[self.e_csum[idx]:self.e_csum[idx+1]]).float()

        n_node = self.n_node[idx:idx+1].astype(int)
        shift = self.shift[self.n_csum[idx]:self.n_csum[idx+1]].astype(float)
        mask = self.mask[self.n_csum[idx]:self.n_csum[idx+1]]
        
        return g, n_node, shift, mask
        
        
    def __len__(self):

        return self.n_node.shape[0]
