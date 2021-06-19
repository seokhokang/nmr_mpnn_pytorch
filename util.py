import torch
import dgl
import numpy as np

                        
def collate_reaction_graphs(batch):

    gs, shifts, masks = map(list, zip(*batch))
    
    gs = dgl.batch(gs)

    shifts = torch.FloatTensor(np.hstack(shifts))
    masks = torch.BoolTensor(np.hstack(masks))
    
    return gs, shifts, masks


def MC_dropout(model):

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
    pass