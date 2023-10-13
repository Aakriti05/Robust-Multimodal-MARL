import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Clonable:
    def clone(self, requires_grad=False):
        clone = copy.deepcopy(self)
        for param in clone.parameters():
            param.requires_grad = requires_grad
        return clone


class Actor(nn.Module, Clonable):
    @classmethod
    def from_actor(cls, actor):
        return cls(actor.n_inputs, actor.action_split, actor.n_hidden)

    def __init__(self, n_inputs, action_split, n_hidden):
        super().__init__()
        self.action_split = tuple(action_split)
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = int(sum(action_split))
        self.lin_1 = nn.Linear(n_inputs, n_hidden)
        self.lin_2 = nn.Linear(n_hidden, n_hidden)
        self.lin_3 = nn.Linear(n_hidden, self.n_outputs)

    def forward(self, x):
        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        logits = self.lin_3(x)
        return logits

    def prob_dists(self, obs, temperature=1.0):
        logits = self.forward(obs)
        split_logits = torch.split(logits, self.action_split, dim=-1)
        temperature = torch.tensor(temperature).to(DEVICE)
        return [RelaxedOneHotCategorical(temperature, logits=l) for l in split_logits]

    def select_action(self, obs, explore=False, temp=1.0):
        distributions = self.prob_dists(obs, temp)
        if explore:
            actions = [d.rsample() for d in distributions]
        else:
            actions = [d.probs for d in distributions]
        return torch.cat(actions, dim=-1)


class Critic(nn.Module, Clonable):
    def __init__(self, n_inputs, n_hidden):
        super().__init__()
        self.n_inputs = n_inputs
        self.lin_1 = nn.Linear(n_inputs, n_hidden)
        self.lin_2 = nn.Linear(n_hidden, n_hidden)
        self.lin_3 = nn.Linear(n_hidden, 1)

    def forward(self, observations, actions):
        x = torch.cat([*observations, *actions], dim=-1)
        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        return self.lin_3(x)


class NatureActor(nn.Module, Clonable):
    """ Similar to Q function, input obs and actions, output scalar """

    def __init__(self, n_inputs, n_hidden):
        super().__init__()
        self.n_inputs = n_inputs
        self.lin_1 = nn.Linear(n_inputs, n_hidden)
        self.lin_2 = nn.Linear(n_hidden, n_hidden)
        self.lin_3 = nn.Linear(n_hidden, 1)

    def forward(self, observations, actions):
        x = torch.cat([*observations, *actions], dim=-1)
        x = F.relu(self.lin_1(x))
        x = F.relu(self.lin_2(x))
        return self.lin_3(x)
