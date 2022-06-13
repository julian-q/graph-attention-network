from torch_geometric.data import InMemoryDataset, Data
import os
import pandas
import numpy as np
import torch
from scipy.io import loadmat
from networkx import from_numpy_matrix
from torch_geometric.utils import from_networkx
from tqdm import tqdm

class ABIDEDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.root + '/ABIDE_pcp/'

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        pheno = pandas.read_csv(self.raw_dir + 'Phenotypic_V1_0b_preprocessed1.csv')

        for subj_id in tqdm(os.listdir(self.raw_dir + 'cpac/filt_noglobal/')):
            subj_dir = self.raw_dir + 'cpac/filt_noglobal/' + subj_id + '/'
            fc_mat = loadmat(subj_dir + subj_id + '_ho_correlation.mat')['connectivity']
            nx_graph = from_numpy_matrix(fc_mat)
            pyg_data = from_networkx(nx_graph)
            label = pheno.loc[pheno['SUB_ID'] == int(subj_id)]['DX_GROUP'].values - 1

            pyg_data.x = torch.tensor(fc_mat, dtype=torch.float32)
            pyg_data.y = torch.tensor(label, dtype=torch.long)
            data_list.append(pyg_data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



