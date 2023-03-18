import random

import torch
from torch import nn
import torch.nn.functional as F
from Agents import Transition
from collections import deque


class ReplayMemoryTrajectory(object):
    def __init__(self, capacity):
        self.transitions = deque([], maxlen=capacity)
        self.current_trajectory = []

    def new_trajectory(self):
        self.transitions.append(self.current_trajectory)

    def push(self, state):
        """Save a transition"""
        self.current_trajectory.append(state)

    def sample_pairs(self, num_trajectories, samples_in_trajectory):
        trajectories = random.sample(self.transitions, num_trajectories)
        samples = []
        for trajectory in trajectories:
            for i in range(samples_in_trajectory):
                xi, xj = random.sample(range(len(trajectory)), 2)
                y = xi < xj
                samples.append((trajectory[xi], trajectory[xj], y))
        return samples

    def __len__(self):
        return len(self.transitions)


class RAE(nn.Module):
    def __init__(self, observation_dim, embedding_dim, beta):
        super(RAE, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(observation_dim, embedding_dim),
                                       nn.ReLU(),
                                       nn.Linear(embedding_dim, embedding_dim))
        self.precedence_estimator = nn.Linear(2 * embedding_dim, 1)
        self.memory = ReplayMemoryTrajectory(10000)
        self.beta = beta

    def forward(self, obv1, obv2):
        obv1_embedding = self.embedding(obv1)
        obv2_embedding = self.embedding(obv2)
        return torch.sigmoid(self.precedence_estimator(torch.cat((obv1_embedding, obv2_embedding), dim=1)))

    def reward(self, obv1, obv2):
        precedence = self(obv1, obv2)
        if precedence > self.beta:
            return -self.beta
        else:
            return 0

    def update(self, batch_size, device, optimizer, samples_in_trajectory):
        if len(self.memory) < batch_size:
            return
        pairs = self.memory.sample_pairs(batch_size, samples_in_trajectory)

        state = torch.cat(list(zip(*pairs))[0])
        next_state = torch.cat(list(zip(*pairs))[1])

        precedence = self(state, next_state)
        y = torch.tensor(list(zip(*pairs))[-1], dtype=torch.float).to(device)

        criterion = SSL_loss()
        loss = criterion(precedence, y)

        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        optimizer.step()


class SSL_loss(nn.Module):
    def __init__(self):
        super(SSL_loss, self).__init__()

    def forward(self, precedence, y):
        assert precedence.shape[0] == y.shape[0]
        batch_size = precedence.shape[0]

        return torch.sum(-y * torch.log(precedence) - (1-y) * torch.log(1 - precedence)) / batch_size

