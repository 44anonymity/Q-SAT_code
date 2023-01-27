import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GMixer_sig(nn.Module):
    def __init__(self, args):
        super(GMixer_sig, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.fc1 = nn.Sequential(nn.Linear(2 * args.rnn_hidden_dim, args.rnn_hidden_dim), nn.ReLU())
        self.fc2_1 = nn.Linear(args.rnn_hidden_dim, 2)
        self.fc2_2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.fc2_3 = nn.Linear(args.rnn_hidden_dim, 1)
        self.fc2_4 = nn.Linear(args.rnn_hidden_dim, 1)
        self.fc_lambda = nn.Linear(args.rnn_hidden_dim, 1)

    def forward(self, agent_qs, hidden_output):  
        bs = agent_qs.size(0)
        hidden_output = hidden_output.reshape(-1, self.n_agents, self.args.rnn_hidden_dim)  
        N = hidden_output.shape[1]
        assert N == self.n_agents, 'hidden_output dimension error'
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        tau = th.cat([hidden_output.repeat_interleave(N, dim=1), hidden_output.repeat(1, N, 1)], dim=-1)  
        temp = self.fc1(tau)
        wij1 = th.abs(self.fc2_1(temp))
        wij2 = th.abs(self.fc2_2(temp))
        bij1 = self.fc2_3(temp)
        bij2 = self.fc2_4(temp)
        lamb = th.sigmoid(self.fc_lambda(temp))
        Q_concat = th.cat([agent_qs.repeat_interleave(N, dim=-1), agent_qs.repeat(1, 1, N)], dim=1).transpose(2, 1)  
        new_Q_mid = F.elu(th.sum(th.mul(wij1, Q_concat), dim=-1, keepdim=True) + bij1)
        new_Q = th.mul(wij2, new_Q_mid) + bij2
        return th.sum(th.mul(lamb, new_Q), dim=1, keepdim=True).view(bs, -1, 1), th.mean(lamb)



