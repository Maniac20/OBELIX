import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import importlib.util

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
##Loading the environment from the provided obelix.py file. This allows us to create instances of the OBELIX environment for training and evaluation.
def load_obelix(path):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


##This code defines a NoisyLinear layer, which is a linear layer with added noise to its weights and biases.
# This is often used in reinforcement learning to encourage exploration.
# The DDQN_LSTM class defines the architecture of the neural network used for the Double DQN algorithm, which includes a fully connected layer, an LSTM layer, and separate streams for advantage and value estimation. 
# The ReplayBuffer class implements a simple experience replay buffer to store and sample past experiences during training. Finally, the train function orchestrates the training loop, including environment interaction, reward shaping, and model updates.
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.ones(out_features, in_features) * 0.017)

        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(torch.ones(out_features) * 0.017)

    def forward(self, x):
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)

        weight = self.weight_mu + self.weight_sigma * weight_eps
        bias = self.bias_mu + self.bias_sigma * bias_eps

        return torch.nn.functional.linear(x, weight, bias)


##The LSTM network , initlializes the layers and defines the forward pass. 
# The network takes an observation as input, processes it through a fully connected layer and an LSTM layer, and then outputs Q-values for each action using separate streams for advantage and value estimation. 
class DDQN_LSTM(nn.Module):
    def __init__(self, obs_dim=18, hidden=128, n_actions=5):
        super().__init__()

        self.fc = nn.Linear(obs_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)

        self.adv = NoisyLinear(hidden, n_actions)
        self.val = NoisyLinear(hidden, 1)

    def forward(self, x, hidden):
        x = torch.relu(self.fc(x))
        x = x.unsqueeze(1)

        out, hidden = self.lstm(x, hidden)
        out = out.squeeze(1)

        adv = self.adv(out)
        val = self.val(out)

        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q, hidden


##Replay buffer is a common component in reinforcement learning algorithms that allows the agent to store and sample past experiences.
class ReplayBuffer:
    def __init__(self, size=50000):
        self.buffer = []
        self.size = size

    def add(self, exp):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append(exp)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

##Train method 
def train(args):
    gamma=0.99 ##discount factor for future rewards
    batch_size=64 

    best_reward = -1e9 ##minimum reward to compare against for saving the best model
    save_dir = "ddqn_runs"

    OBELIX = load_obelix(args.obelix_py)

    policy_net = DDQN_LSTM()
    target_net = DDQN_LSTM()
    target_net.load_state_dict(policy_net.state_dict())

    # optimizer = optim.Adam(policy_net.parameters(), lr=2e-3)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    

    buffer = ReplayBuffer() ##initlise the  buffer


    os.makedirs(save_dir, exist_ok=True)

    def get_difficulty(ep): ##curriculum learning strategy to increase difficulty as training progresses
        if ep < 200:
            return 1
        elif ep < 500:
            return 2
        else:
            return 3

    for ep in range(args.episodes):

        env = OBELIX(
            difficulty=get_difficulty(ep),
            max_steps=800,
            wall_obstacles=True,
            seed=ep,
            scaling_factor=5,
            arena_size=500,
        )

        obs = env.reset(seed=ep)

        hidden = (
            torch.zeros(1, 1, 128),
            torch.zeros(1, 1, 128)
        )

        total_reward = 0
        done = False

        while not done:

            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            hidden = (hidden[0].detach(), hidden[1].detach())##detach hidden state to prevent backprop through time across episodes

            if obs[17] == 1:
                action = random.choice([0, 4]) ##unwedge
            else:
                q_values, hidden = policy_net(obs_t, hidden)
                action = torch.argmax(q_values).item()

            next_obs, reward, done = env.step(ACTIONS[action], render=False)

            ##reward shhaping 
            if next_obs[16] == 1 and action == 2:
                reward+=8

            if np.any(next_obs[0:4]) and action == 2:
                reward+=3

            if np.sum(next_obs[:17]) == 0 and action == 2:
                reward-=3

            reward = max(reward, -200)

            buffer.add((obs, action, reward, next_obs, done))

            obs = next_obs
            total_reward += reward
            
            ##train the network if we have enough samples in the buffer

            if len(buffer) > batch_size:

                s, a, r, ns, d = buffer.sample(batch_size)

                s=torch.tensor(s, dtype=torch.float32)
                ns=torch.tensor(ns, dtype=torch.float32)
                a=torch.tensor(a, dtype=torch.long)
                r=torch.tensor(r, dtype=torch.float32)
                d=torch.tensor(d, dtype=torch.float32)

                # fresh hidden
                h0=torch.zeros(1, batch_size, 128)
                c0=torch.zeros(1, batch_size, 128)
                h=(h0, c0)

                q_values, _ = policy_net(s, h)
                q_values = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_policy, _ = policy_net(ns, h)
                    next_actions = torch.argmax(next_q_policy, dim=1)

                    next_q_target, _ = target_net(ns, h)
                    next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

                    target = r + gamma * next_q * (1 - d)

                loss = ((q_values - target) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)

                optimizer.step()
                
        #update target network every 20 episodes

        if ep % 20 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        ##save the best one here

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy_net.state_dict(),
                       f"{save_dir}/weight_best_{int(best_reward)}.pth")

        ##logging for the plot 
        if (ep + 1) % 10 == 0:
            print(f"Ep {ep+1} | Reward: {total_reward:.1f} | Best: {best_reward:.1f}")

            with open(f"{save_dir}/log.txt", "a") as f:
                f.write(f"{ep+1},{total_reward},{best_reward}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1000)

    args = parser.parse_args()
    train(args)