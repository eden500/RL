from collections import namedtuple, deque, defaultdict
import random
from itertools import count
import torch.nn.functional as F
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from torch import nn
import wandb
import matplotlib.cm as cm


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, observation_dim, action_dim, embedding_dim, memory_size=200000):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(observation_dim, embedding_dim)
        self.layer2 = nn.Linear(embedding_dim, embedding_dim)
        self.layer3 = nn.Linear(embedding_dim, action_dim)
        self.memory = ReplayMemory(memory_size)
        self.episode_durations = []

    def forward(self, obv):
        obv = F.relu(self.layer1(obv))
        obv = F.relu(self.layer2(obv))
        return self.layer3(obv)

    def select_action(self, state, random_action=False, env=None):
        if random_action:
            if env is None:
                raise ValueError("Need to give environment when random action is on")
            return torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=state.device)
        else:
            with torch.no_grad():
                return self(state).max(1)[1].view(1, 1)

    def update(self, target_agent, batch_size, device, agent, discount_factor, optimizer):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = agent(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_agent(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * discount_factor) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(agent.parameters(), 100)
        optimizer.step()


def train(num_episodes, env, agent, target_agent, device, batch_size, discount_factor, optimizer, rae_model=None,
          rae_optimizer=None, seed=0, test_rae=False, wandb_log=True, update_plot=True):
    # plt.ion()
    if wandb_log:
        wandb.init(
            project="reversibilityRL",

            config={
                "seed": seed,
                "rae_model": bool(rae_model)
            })

    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            sample = random.random()
            eps_threshold = 0.05 + np.exp(-1. * i_episode / 100)
            chose_random = sample < eps_threshold
            action = agent.select_action(state, chose_random, env)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([0], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                if rae_model:
                    with torch.no_grad():
                        reversibility_reward = rae_model.reward(state, next_state)
                        reward += reversibility_reward

                    rae_model.memory.push(state)

            agent.memory.push(state, action, next_state, reward)
            state = next_state
            agent.update(target_agent, batch_size, device, agent, discount_factor, optimizer)

            target_net_state_dict = target_agent.state_dict()
            policy_net_state_dict = agent.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * 0.005 + target_net_state_dict[key] * (
                        1 - 0.005)
            target_agent.load_state_dict(target_net_state_dict)

            if done:
                if rae_model:
                    rae_model.memory.new_trajectory()
                agent.episode_durations.append(t + 1)
                if wandb_log:
                    wandb.log({"episode_duration": t + 1})
                if update_plot:
                    plot_durations(agent)
                else:
                    print("Episode {} finished after {} timesteps".format(i_episode, t + 1))
                break

        if rae_model:
            rae_model.update(3, device, rae_optimizer, 3)

        if test_rae:
            if i_episode in [100, 500, 1000, 3000]:
                plot_rae(rae_model, agent.memory.memory)

    print('Complete')
    plt.ioff()
    if update_plot:
        plot_durations(agent, show_result=True)
        plt.show()

    wandb.finish()

    return agent.episode_durations


def plot_rae(rae_model, replay_memory):
    right_dictionary = defaultdict(list)
    left_dictionary = defaultdict(list)
    rae_dictionary = defaultdict(list)
    for transition in replay_memory:
        if transition.next_state is None:
            continue
        with torch.no_grad():
            rae_dictionary[transition.state[0][2]].append(rae_model(transition.state, transition.next_state).item())
            if transition.action == 0:
                left_dictionary[transition.state[0][2]].append(rae_model(transition.state, transition.next_state).item())
            else:
                right_dictionary[transition.state[0][2]].append(rae_model(transition.state, transition.next_state).item())

    def polar(angle):
        return np.cos(angle), np.sin(angle)

    left_dict = {}
    for key, value in left_dictionary.items():
        left_dict[polar(round(key.cpu().item(), 3))] = np.mean(value)
    right_dict = {}
    for key, value in right_dictionary.items():
        right_dict[polar(round(key.cpu().item(), 3))] = np.mean(value)
    rae_dict = {}
    for key, value in rae_dictionary.items():
        rae_dict[polar(round(key.cpu().item(), 3))] = np.mean(value)


    plt.scatter([x[1] for x in left_dict.keys()], [x[0] for x in left_dict.keys()], c=list(left_dict.values()), cmap='RdBu')
    plt.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap='RdBu'))
    plt.savefig(f'rae_left_{len(replay_memory)}.png')
    plt.clf()

    plt.scatter([x[1] for x in right_dict.keys()], [x[0] for x in right_dict.keys()], c=list(right_dict.values()), cmap='RdBu')
    plt.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap='RdBu'))
    plt.savefig(f'rae_right_{len(replay_memory)}.png')
    plt.clf()

    plt.scatter([x[1] for x in rae_dict.keys()], [x[0] for x in rae_dict.keys()], c=list(rae_dict.values()), cmap='RdBu')
    plt.colorbar(cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap='RdBu'))
    plt.savefig(f'rae_{len(replay_memory)}.png')
    plt.clf()


def plot_durations(agent, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(agent.episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
