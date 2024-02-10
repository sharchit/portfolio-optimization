from gymenv import Portfolio

import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(2024)
torch.cuda.manual_seed(2024)
torch.manual_seed(2024)

state = pd.read_csv('../data/rl/states.csv')
price = pd.read_csv('../data/rl/price.csv')
action = np.load('../data/rl/actions.np.npy')

env = Portfolio(price.values, action, state.values)
env.reset()

s_size = env.observation_space.shape[0]
a_size = env.action_space.n

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

Gamma = 0.9
Episodes = 1000

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(s_size, 128)
        self.attn1 = nn.Linear(s_size, s_size)

        self.ac1 = nn.Linear(128, 128)
        self.acrel = nn.ReLU()
        self.action_head = nn.Linear(128, a_size)

        self.cr1 = nn.Linear(128, 128)
        self.crrel = nn.ReLU()
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        attn = F.sigmoid(self.attn1(x))
        x = torch.mul(x, attn)
        x = F.relu(self.affine1(x))

        y = self.ac1(x)
        y = self.acrel(y)
        action_prob = F.softmax(self.action_head(y), dim=-1)

        z = self.cr1(x)
        z = self.crrel(z)
        state_values = self.value_head(z)

        return action_prob, state_values, attn
    
model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value, attn = model(state)

    m = Categorical(probs)

    action = m.sample()

    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    return action.item(), attn

def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    value_losses = []
    policy_losses = []
    returns = []

    for r in model.rewards[::-1]:
        R = r + Gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)

        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    optimizer.zero_grad()

    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]

if __name__ == '__main__':

    #for i_episode in trange(2): # Change to Episodes
    for i_episode in trange(Episodes):
        state, _ = env.reset()
        ep_reward = 0
        cnt = 0
        for t in range(1, 10000):
            action, attn = select_action(state)
            state, reward, done, _, _ = env.step({'action': action, 'attn': attn.detach().numpy()})

            model.rewards.append(reward)
            ep_reward += reward
            cnt+=1
            if done:
                break

        
        writer.add_scalar("Loss/train", ep_reward/cnt, i_episode)
        writer.flush()
        
        finish_episode()
        
print('Training Complete')
torch.save(model.state_dict(), './a2c-policy2.pt')
