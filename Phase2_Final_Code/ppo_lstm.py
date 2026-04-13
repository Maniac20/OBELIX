import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import argparse
import importlib.util

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


def load_obelix(path):
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


##Apply PPO with LSTM to solve the Obelix environment
class PPO_LSTM(nn.Module):
    def __init__(self, obs_dim=18, hidden=128, n_actions=5):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(hidden, hidden)

        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x, hidden):
        x = self.fc(x)
        x = x.unsqueeze(0).unsqueeze(0)

        out, hidden = self.lstm(x, hidden)
        out = out.squeeze(0).squeeze(0)

        logits = self.actor(out)
        value = self.critic(out)

        return logits, value.squeeze(-1), hidden

    def get_action(self, obs, hidden):
        logits, value, hidden = self.forward(obs, hidden)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value, hidden

    def evaluate(self, obs_seq, actions, hidden):
        log_probs, values, entropies = [], [], []

        for t in range(len(obs_seq)):
            logits, value, hidden = self.forward(obs_seq[t], hidden)
            dist = Categorical(logits=logits)

            log_probs.append(dist.log_prob(actions[t]))
            entropies.append(dist.entropy())
            values.append(value)

        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)


##Train the PPO agent with LSTM on the Obelix environment
def train(args):

    OBELIX = load_obelix(args.obelix_py)

    agent = PPO_LSTM()
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)

    gamma = 0.99
    lam = 0.95
    clip_eps = 0.2
    epochs = 4

    best_reward = -1e9

    save_dir = "ppo_lstm_weights"
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(args.episodes):

        env = OBELIX(
            difficulty=2,
            max_steps=800,   #keep it less here ,we will do early termination if it goes too bad
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

        obs_list, actions, log_probs = [], [], []
        rewards, values, dones = [], [], []

        total_reward = 0
        done = False
        stuck_counter = 0

        while not done:

            obs_t = torch.tensor(obs, dtype=torch.float32)
            
            ##unwedge logic, if STUCK bit is set, take random action for a few steps to break the loop

        
            is_stuck = obs[17] == 1  # Check if the STUCK bit is set    

            if is_stuck:
                stuck_counter += 1
                action_idx = np.random.choice([0, 4])  # Randomly choose between L45 and R45 to try to break the loop
                action = torch.tensor(action_idx)
                log_prob = torch.tensor(0.0)
                entropy = torch.tensor(0.0)
                value = torch.tensor(0.0)
            else:
                stuck_counter = 0
                action, log_prob, entropy, value, hidden = agent.get_action(obs_t, hidden)

            next_obs, reward, done = env.step(ACTIONS[action.item()], render=False)
            
            ##custom reward shaping to encourage better learning and break loops


            if next_obs[16] == 1 and action.item() == 2:
                reward += 8

            if np.any(next_obs[0:4]) and action.item() == 2:
                reward += 3

            if np.any(next_obs[4:6]) and action.item() in [0,1]:
                reward += 2

            if np.any(next_obs[6:8]) and action.item() in [3,4]:
                reward += 2

            if np.sum(next_obs[:17]) == 0 and action.item() == 2:
                reward -= 3

            if np.sum(next_obs[:17]) == 0 and action.item() in [0,1,3,4]:
                reward += 0.5

            ##clipping large negative rewards to prevent destabilization and encourage learning from bad states rather than just giving up
    
            reward = max(reward, -200)

            ##early termination if total reward goes too bad to save time and encourage learning from better trajectories
       
            total_reward += reward
            if total_reward < -3000:
                done = True

            obs_list.append(obs_t)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

            obs = next_obs
            
        ##gae

   
        advantages, returns = [], []
        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae

            advantages.insert(0, gae)
            next_value = values[t]

        for t in range(len(rewards)):
            returns.append(advantages[t] + values[t])

        obs_tensor = torch.stack(obs_list)
        actions_tensor = torch.stack(actions)
        old_log_probs = torch.stack(log_probs).detach()

        advantages = torch.stack(advantages).detach()
        returns = torch.stack(returns).detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ##Ppo update with multiple epochs and early stopping if the reward is good enough to save time and encourage learning from better trajectories
        for _ in range(epochs):

            hidden = (
                torch.zeros(1, 1, 128),
                torch.zeros(1, 1, 128)
            )

            new_log_probs, entropy, new_values = agent.evaluate(
                obs_tensor, actions_tensor, hidden
            )

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - new_values).pow(2).mean()
            entropy_loss = entropy.mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            
            
        ##save the best score model based on total reward to ensure we have a good performing model for submission and encourage learning from better trajectories


        if total_reward > best_reward:
            best_reward = total_reward

            save_path = os.path.join(
                save_dir,
                f"weight_best_{int(best_reward)}.pth"
            )
            torch.save(agent.state_dict(), save_path)

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1} | Reward: {total_reward:.1f} | Best: {best_reward:.1f}")

    torch.save(agent.state_dict(), os.path.join(save_dir, "weights_final.pth"))
    print("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obelix_py", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    args = parser.parse_args()
    train(args)