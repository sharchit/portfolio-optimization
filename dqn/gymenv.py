import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class Portfolio(gym.Env):
    def __init__(self, price_matrix, action_matrix, state_matrix, A=80000, B=0.5, TC=0.006):
        assert len(price_matrix) == len(state_matrix)
        self.A = A
        self.B = B
        self.TC = TC
        self.price_matrix = price_matrix
        self.state_matrix = state_matrix
        self.action_matrix = action_matrix
        self.done = False
        self.old_val = self.po_value(np.ones(10) / 10, self.price_matrix[0])
        self.index = 0
        self.action_space = spaces.Discrete(500)
        self.observation_space = spaces.Box(low=np.zeros(64), high=np.ones(64) * 1_000_000, dtype=np.float64)

    def reset(self, seed=None):
        self.index = 1
        self.done = False
        self.old_val = self.po_value(np.zeros(10), self.price_matrix[0])
        return self.state_matrix[self.index -1], {'done': self.done}

    def po_value(self, action, price):
        return np.dot(action, price)

    def step(self, action_attn):
        action, attn = action_attn['action'], action_attn['attn']
        
        attention_loss = ((attn - 0.5*np.ones(64))**2).sum()
        
        self.curr = self.po_value(self.action_matrix[action], (self.price_matrix[self.index] - self.price_matrix[self.index - 1]) / self.price_matrix[self.index - 1])
        
        reward = self.A * (self.curr) + self.B * (attention_loss) + self.A * self.TC * (self.curr - self.old_val)
        
        self.index += 1
        done = self.index >= len(self.price_matrix) - 1
        obs = self.state_matrix[self.index]
        self.old_val = self.curr
        return obs, reward, done, False, {'i': self.index}

# state = pd.read_csv('./data/rl/states.csv')
# price = pd.read_csv('./data/rl/price.csv')
# action = np.load('./data/rl/actions.np.npy')

# env = Portfolio(price.values, action, state.values, 20000, 1)

# state, info = env.reset()
# done = info['done']
# rew = []
# while not done:
#     action = env.action_space.sample()
#     obs, reward, done, _, _ = env.step({'action': action, 'attn': np.random.rand(64)})
#     rew.append(reward)

# print(sum(rew)/len(rew))
# print(len(rew))
