import torch
import torch.nn as nn
import torch.nn.functional as F


class multvae():
    def __init__(self,n_users,n_items,original_dim,hidden_dim,
                 latent_dim,n_epochs,k,drop_encoder,
                 drop_decoder,beta,annealing,anneal_cap,random_state):
        self.n_users = n_users
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        pass
    
    def encoder(self):
        layers = []
        current_dim = self.n_items
        for h_dim in self.hidden_dim:
            layers.append(nn.Linear(current_dim,h_dim))
            layers.append(nn.Tanh())
            current_dim = h_dim
        self.mlp = nn.Sequential(*layers)
        
        pass