import numpy as np
import pandas as pd

from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from gymenv import Portfolio

from tqdm import trange

from torch.utils.tensorboard import SummaryWriter

np.random.seed(2024)
torch.cuda.manual_seed(2024)
torch.manual_seed(2024)

writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

state = pd.read_csv('../data/rl/states.csv')
price = pd.read_csv('../data/rl/price.csv')

env = Portfolio(price.values, state.values)

s_size = env.observation_space.shape[0]
a_size = env.action_space.shape[0]
print(a_size)
class Policy(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.attn1 = nn.Linear(n_observations, n_observations)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        attn = F.sigmoid(self.attn1(x))
        x = torch.mul(x, attn)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        mean = self.layer3(x)
        std = torch.abs(self.layer4(x))
        return mean, std, attn

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mean, std, attn = self.forward(state)
        m = Normal(mean, std)
        action = m.sample()
        return action, m.log_prob(action), attn
    
def reinforce(policy, optimizer, n_training_episodes, max_t, gamma):
    scores_deque = deque(maxlen=4000)
    scores = []

    for i_episode in trange(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()

        for t in range(max_t):
            action, log_prob, attn = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ , _= env.step({'action': action.detach(), 'attn': attn.detach()})
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        returns = deque(maxlen=max_t)
        n_steps = len(rewards)

        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[t])

        eps = np.finfo(np.float32).eps.item()

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
    
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", np.mean(scores_deque), i_episode)
        writer.flush()
        # print("Episode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_deque)))
    return scores

hyperparameters = {
    "n_training_episodes": 1000,
    "max_t": 4000,
    "gamma": 1,
    "lr": 1e-4,
    "state_space": s_size,
    "action_space": a_size,
}

policy = Policy(hyperparameters["state_space"], hyperparameters["action_space"],).to(device)
optimizer = optim.Adam(policy.parameters(), lr=hyperparameters["lr"])

scores = reinforce(
    policy,
    optimizer,
    hyperparameters["n_training_episodes"],
    hyperparameters["max_t"],
    hyperparameters["gamma"],
)

print('Training Complete')
torch.save(policy.state_dict(), './reinforced-policy2.pt')
