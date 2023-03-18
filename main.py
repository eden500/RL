import gym
import torch.optim
import numpy as np
import random
import matplotlib.pyplot as plt
from Agents import DQN, train
from ReversibilityAwareAlgs import RAE
import wandb


def main():
    for seed in range(10):
        set_seed(seed)
        # env = gym.make('CartPole-v1', render_mode='human')
        env = gym.make('CartPole-v1')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        agent = DQN(observation_dim=env.observation_space.shape[0], action_dim=env.action_space.n, embedding_dim=64).to(
            device)
        target_agent = DQN(observation_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
                           embedding_dim=64).to(device)
        target_agent.load_state_dict(agent.state_dict())

        optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-4, amsgrad=True)

        rae_model = RAE(env.observation_space.shape[0], 64, beta=0.8).to(device)
        rae_optimizer = torch.optim.Adam(rae_model.parameters(), lr=1e-3)

        rae_model = None
        rae_optimizer = None

        results = train(1000, env, agent, target_agent, device, batch_size=128, discount_factor=0.99, optimizer=optimizer,
                        rae_model=rae_model, rae_optimizer=rae_optimizer, seed=seed, test_rae=False, wandb_log=True, update_plot=False)

        print(results)
        with open("results.csv", 'a') as f:
            f.write(str(bool(rae_model)) + ',' + str(seed) + ',' + ' '.join([str(x) for x in results]))
            f.write('\n')


def test_env(env):
    for _ in range(100):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated:  # or truncated:
            observation, info = env.reset()
    env.close()


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    main()
