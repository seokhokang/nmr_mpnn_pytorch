import numpy as np
import torch
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset

from dataset import GraphDataset
from util import collate_reaction_graphs
from model import nmrMPNN, training, inference

data_split = [0.8, 0.1, 0.1]
batch_size = 32
use_pretrain = False
model_path = './nmr_model.pt'
random_seed = 1

data = GraphDataset()
node_dim = data.node_attr.shape[1]
edge_dim = data.edge_attr.shape[1]

train_set, val_set, test_set = split_dataset(data, data_split, shuffle=True, random_state=random_seed)

train_y = np.hstack([data[idx][1] for idx in train_set.indices])
train_y_mean = np.mean(train_y)
train_y_std = np.std(train_y)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_reaction_graphs, drop_last=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size * 10, shuffle=False, collate_fn=collate_reaction_graphs)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size * 10, shuffle=False, collate_fn=collate_reaction_graphs)

net = nmrMPNN(node_dim, edge_dim).cuda()

print('--- configurations')
print('--- data_size:', data.__len__())
print('--- train/val/test: %d/%d/%d' %(train_set.__len__(), val_set.__len__(), test_set.__len__()))
print('--- use_pretrain:', use_pretrain)
print('--- model_path:', model_path)


# training
if use_pretrain == False: net = training(net, train_loader, val_loader, train_y_mean, train_y_std, model_path)

# inference
net.load_state_dict(torch.load(model_path))
tsty_pred = inference(net, test_loader, train_y_mean, train_y_std)

tsty = np.hstack([inst[1][inst[2]] for inst in iter(test_loader.dataset)])

print('--- test MAE', np.mean(np.abs(tsty - tsty_pred)))