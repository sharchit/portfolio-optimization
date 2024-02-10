import os
from gymenv import Portfolio
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import pandas as pd
import numpy as np
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 3e-4
episodes = 1000

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.attn1 = nn.Linear(self.state_size, self.state_size)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        attn = F.sigmoid(self.attn1(state))
        state = torch.mul(state, attn)
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution, attn


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for iter in trange(n_iters):
        state, _ = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()

        for i in count():
            state = torch.FloatTensor(state).to(device)
            dist, attn = actor(state)
            value = critic(state)

            action = dist.sample()
            next_state, reward, done, _, _ = env.step({'action': action, 'attn': attn.detach().numpy()})

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state

            if done:
                writer.add_scalar("Loss/train", (sum(rewards)/len(rewards)).item(), iter)
                writer.flush()
                break


        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    print('Training Complete')
    torch.save(actor.state_dict(), 'actor.pt')
    torch.save(critic.state_dict(), 'critic.pt')
    env.close()


if __name__ == '__main__':
    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size, action_size).to(device)
    trainIters(actor, critic, n_iters=episodes)