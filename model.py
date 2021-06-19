import numpy as np
import time

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dgl.nn.pytorch import NNConv

from util import MC_dropout

class nmrMPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats,
                 node_hidden_feats = 64, node_out_feats = 64,
                 edge_hidden_feats = 128,
                 num_step_message_passing = 6,
                 num_step_set2set = 6, num_layer_set2set = 3,
                 predict_hidden_feats = 512, prob_dropout = 0.1):
        
        super(nmrMPNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hidden_feats), nn.ReLU(),
            nn.Linear(node_hidden_feats, node_hidden_feats), nn.ReLU(),
            nn.Linear(node_hidden_feats, node_hidden_feats), nn.ReLU(),
            nn.Linear(node_hidden_feats, node_out_feats), nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats), nn.ReLU(),
            nn.Linear(edge_hidden_feats, edge_hidden_feats), nn.ReLU(),
            nn.Linear(edge_hidden_feats, edge_hidden_feats), nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)
        )
        self.gnn_layer = NNConv(
            in_feats = node_out_feats,
            out_feats = node_out_feats,
            edge_func = edge_network,
            aggregator_type = 'sum'
        )
        self.gru = nn.GRU(node_out_feats, node_out_feats)
                               
        self.predict = nn.Sequential(
            nn.Linear(node_out_feats, predict_hidden_feats), nn.ReLU(),
            nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, predict_hidden_feats), nn.ReLU(),
            nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, 1)
        )                           
                               
    def forward(self, g):
        
        def embed(g):
            
            node_feats = g.ndata['attr']
            edge_feats = g.edata['edge_attr']
            
            node_feats = self.project_node_feats(node_feats)
            hidden_feats = node_feats.unsqueeze(0)

            for _ in range(self.num_step_message_passing):
                node_feats = torch.relu(self.gnn_layer(g, node_feats, edge_feats))
                node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
                node_feats = node_feats.squeeze(0)
            
            return node_feats

        node_embed_feats = embed(g)
        out = self.predict(node_embed_feats).flatten()

        return out

        
def training(net, train_loader, val_loader, model_path, max_epochs = 500, print_intv = 1000, n_forward_pass = 5):

    cuda = torch.device('cuda:0')
    
    train_size = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size
    trn_shift_std = np.std(np.hstack([inst[1][inst[2]] for inst in iter(train_loader.dataset)]))

    loss_fn = nn.L1Loss()
    optimizer = Adam(net.parameters(), lr=1e-3)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6, verbose=True)

    val_log = np.zeros(max_epochs)
    for epoch in range(max_epochs):
        
        # training
        net.train()
        start_time = time.time()
        for batchidx, batchdata in enumerate(train_loader):

            inputs = batchdata[0].to(cuda)
            shifts = batchdata[1].to(cuda)
            masks = batchdata[2].to(cuda)
            
            shifts = shifts[masks]
            predictions = net(inputs)[masks]

            loss = loss_fn(predictions, shifts) / trn_shift_std 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (1 + batchidx) % print_intv == 0:
                print('--- training epoch %d, processed %d/%d, loss %.3f, time elapsed(min) %.2f' %(epoch,  batch_size * (1 + batchidx), train_size, loss.detach().item() * trn_shift_std, (time.time()-start_time)/60))

        print('--- training epoch %d, processed %d/%d, loss %.3f, time elapsed(min) %.2f' %(epoch,  train_size, train_size, loss.detach().item() * trn_shift_std, (time.time()-start_time)/60))
    
        # validation
        net.eval()
        MC_dropout(net)
        val_loss, val_cnt = 0, 0
        with torch.no_grad():
            for batchidx, batchdata in enumerate(val_loader):
            
                inputs = batchdata[0].to(cuda)
                shifts = batchdata[1].numpy()
                masks = batchdata[2].numpy()
                
                shifts = shifts[masks]
                predictions_list = [net(inputs).cpu().numpy()[masks] for _ in range(n_forward_pass)]
                predictions = np.mean(predictions_list, 0)
    
                loss = np.abs(shifts - predictions)
                
                val_loss += np.sum(loss)
                val_cnt += len(loss)
    
        val_loss = val_loss/val_cnt
        
        val_log[epoch] = val_loss
        print('--- validation, processed %d, current MAE %.3f, best MAE %.3f' %(val_loader.dataset.__len__(), val_loss, np.min(val_log[:epoch + 1])))
        
        lr_scheduler.step(val_loss)
        
        # earlystopping
        if np.argmin(val_log[:epoch + 1]) == epoch:
            torch.save(net.state_dict(), model_path) 
        
        elif np.argmin(val_log[:epoch + 1]) <= epoch - 20:
            break

    print('training terminated at epoch %d' %epoch)
    net.load_state_dict(torch.load(model_path))
    
    return net
    

def inference(net, test_loader, n_forward_pass = 30):
    
    cuda = torch.device('cuda:0')
    
    net.eval()
    MC_dropout(net)
    tsty_pred = []
    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):
        
            inputs = batchdata[0].to(cuda)
            masks = batchdata[2].numpy()

            tsty_pred.append(np.array([net(inputs).cpu().numpy()[masks] for _ in range(n_forward_pass)]).transpose())

    tsty_pred = np.vstack(tsty_pred)
    
    return np.mean(tsty_pred, 1)
