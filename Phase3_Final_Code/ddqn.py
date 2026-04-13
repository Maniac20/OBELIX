import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import argparse
import importlib.util

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

##load the obelix environment from the provided path
def load_obelix(path):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX



##Q network 
class QNet(nn.Module):
    def __init__(self, obs_dim=18, hidden=128, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 3, hidden),  # stacking 3 frames
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)


##replay buffer for experience replay in DQN
class ReplayBuffer:
    def __init__(self, size=100000):
        self.buffer = deque(maxlen=size)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)


##my heuristic action function to provide some guidance to the agent and encourage learning from better trajectories
def heuristic_action(obs):
    if obs[17] == 1: ##if attached push forward to encourage learning from good trajectories
        return 2 

    if np.sum(obs[:17]) == 0:
        return random.choice([0, 1, 3, 4]) ##if no signal, roate and explore to encourage learning from better trajectories

    
    if np.any(obs[0:4]): ##if front and then go forward 
        return 2

    if np.any(obs[4:6]):  ##left singal
        return random.choice([0, 1])

    # right signal
    if np.any(obs[6:8]):
        return random.choice([3, 4])

    return None

##training state construction with frame stacking to provide temporal context to the agent and encourage learning from better trajectories
def train(args):

    OBELIX = load_obelix(args.obelix_py)

    device = torch.device("cpu")

    q_net = QNet().to(device)
    target_net = QNet().to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

    buffer = ReplayBuffer()

    gamma = 0.99
    batch_size = 64

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    best_reward = -1e9

    for ep in range(args.episodes):

        env = OBELIX(
            difficulty=3,
            max_steps=2000,
            wall_obstacles=True,
            scaling_factor=5,
            seed=ep
        )

        obs = env.reset(seed=ep)

        prev_obs = [np.zeros_like(obs), np.zeros_like(obs)] ##frame stacking with two previous observations to provide temporal context

        total_reward = 0
        done = False

        while not done:

            state = np.concatenate([obs, prev_obs[0], prev_obs[1]])
            state_t = torch.tensor(state, dtype=torch.float32).to(device)

            h_act = heuristic_action(obs) ##override action with heuristic action to encourage learning from better trajectories

            if h_act is not None and random.random() < 0.7:
                action = h_act
            else:
                if random.random() < epsilon:
                    action = random.randint(0, 4)
                else:
                    with torch.no_grad():
                        q_vals = q_net(state_t)
                        action = torch.argmax(q_vals).item()

            next_obs, reward, done = env.step(ACTIONS[action], render=False)

            if reward == -200: ##shape 
                reward -= 50  # strong wall penalty

            next_state = np.concatenate([next_obs, obs, prev_obs[0]])

            buffer.push(state, action, reward, next_state, done)

            prev_obs = [obs, prev_obs[0]]
            obs = next_obs
            total_reward += reward

            # learn
            if len(buffer) > batch_size:
                s, a, r, ns, d = buffer.sample(batch_size)

                s = torch.tensor(s, dtype=torch.float32)
                a = torch.tensor(a)
                r = torch.tensor(r, dtype=torch.float32)
                ns = torch.tensor(ns, dtype=torch.float32)
                d = torch.tensor(d, dtype=torch.float32)

                q = q_net(s).gather(1, a.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_q = target_net(ns).max(1)[0]
                    target = r + gamma * next_q * (1 - d)

                loss = (q - target).pow(2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if ep % 10 == 0:
            target_net.load_state_dict(q_net.state_dict()) ##update the target 

        epsilon = max(epsilon_min, epsilon * epsilon_decay) ##decay the epsilon to encourage more exploitation over time

        if total_reward > best_reward: ##save every best reward 
            best_reward = total_reward
            torch.save(q_net.state_dict(), f"ddqn_runs/{best_reward:.2f}_best_weights.pth")

        if (ep + 1) % 10 == 0:
            print(f"Ep {ep+1} | Reward: {total_reward:.1f} | Best: {best_reward:.1f} | Eps: {epsilon:.2f}")

    torch.save(q_net.state_dict(), "ddqn_runs/final_weights.pth")
    print("Saved best_weights.pth and final_weights.pth")


##main method
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1500)

    args = parser.parse_args()
    train(args)