import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .mix_attention import Mixer_attention
import numpy as np
class GMixer_sig_atten_mlp(nn.Module):
    def __init__(self, args):
        super(GMixer_sig_atten_mlp, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.atten_1 = Mixer_attention(args)
        
        self.state_dim = int(np.prod(args.state_shape))        
        self.embed_dim = args.mixing_embed_dim
        hypernet_embed = self.args.hypernet_embed
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.n_agents))
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),        
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, hidden_output, states):  
        bs = agent_qs.size(0)
        hidden_output = hidden_output.reshape(-1, self.n_agents, self.args.rnn_hidden_dim) 
        agent_qs = agent_qs.view(-1, 1, self.n_agents) # (batch,1,N)

        new_Q_i_1 , l1_loss = self.atten_1(agent_qs, hidden_output)# (batch,1,N)
        
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.n_agents, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(new_Q_i_1, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot, l1_loss