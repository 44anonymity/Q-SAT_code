import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
from .qatten_weight import Qatten_Weight
from .dmaq_si_weight import DMAQ_SI_Weight
from .mix_attention import Mixer_attention

class DMAQ_QattenMixer_Atten(nn.Module):
    def __init__(self, args):
        super(DMAQ_QattenMixer_Atten, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1

        self.attention_weight = Qatten_Weight(args)
        self.si_weight = DMAQ_SI_Weight(args)
        self.atten_1 = Mixer_attention(args)

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = th.sum(agent_qs, dim=-1)
        return v_tot

    def calc_adv(self, agent_qs, states, actions, max_q_i):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        adv_w_final = self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        if self.args.is_minus_one:
            adv_tot = th.sum(adv_q * (adv_w_final - 1.), dim=1)
        else:
            adv_tot = th.sum(adv_q * adv_w_final, dim=1)
        return adv_tot

    def calc(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_q_i)
            return adv_tot

    def forward(self, agent_qs, hidden_output, states, actions=None, max_q_i=None, is_v=False):
        bs = agent_qs.size(0)
        # print('agent_qs', agent_qs.shape)
        hidden_output = hidden_output.reshape(-1, self.n_agents, self.args.rnn_hidden_dim)    # (batch*T,N,D)
        agent_qs = agent_qs.view(-1, 1, self.n_agents) # (batch,1,N)

        new_Q_i , l1_loss = self.atten_1(agent_qs, hidden_output)# (batch,1,N)
        # print('new_Q_i', new_Q_i.shape)
        new_Q_i = new_Q_i.view(bs, -1, self.n_agents)
        
        w_final, v, attend_mag_regs, head_entropies = self.attention_weight(new_Q_i, states, actions)
        w_final = w_final.view(-1, self.n_agents)  + 1e-10
        v = v.view(-1, 1).repeat(1, self.n_agents)
        v /= self.n_agents

        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w_final * agent_qs + v
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)
            max_q_i = w_final * max_q_i + v

        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        return v_tot, attend_mag_regs, head_entropies, l1_loss
