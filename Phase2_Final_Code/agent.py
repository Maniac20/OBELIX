from __future__ import annotations
from typing import Optional, List
import os
import numpy as np
import torch
import torch.nn as nn


ACTIONS:    List[str]=["L45" ,"L22" ,"FW" ,"R22" ,"R45"]


##thisis my lstm and ppo model 
class PPOAgent(nn.Module):
    def __init__(self, obs_dim=18, hidden=128, n_actions=5):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(hidden, hidden)

        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)  ##this is  required for loading 

    def forward(self, x, hidden):
        x = self.fc(x)
        x = x.unsqueeze(0).unsqueeze(0)  

        out, hidden = self.lstm(x, hidden)
        out = out.squeeze(0).squeeze(0)

        logits = self.actor(out)
        return logits, hidden


##global variables to hold the model and hidden state across calls
_hidden = None

_model: Optional[PPOAgent] = None




def _load_once():
    global _model
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")

    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found next to agent.py")

    model = PPOAgent()

    sd = torch.load(wpath, map_location="cpu")

    # handle wrapped checkpoints
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model.load_state_dict(sd, strict=True)
    model.eval()

    _model = model


_last_action = None
_repeat = 0

@torch.no_grad()
def policy(obs, rng):
    global _hidden, _last_action, _repeat

    _load_once()

    if obs[17] == 1:
        return rng.choice(["L45", "R45"]) ##unwedhing the agent from the loop

    if _hidden is None:
        _hidden = (
            torch.zeros(1, 1, 128),
            torch.zeros(1, 1, 128)
        )

    x = torch.tensor(obs, dtype=torch.float32)
    logits, _hidden = _model(x, _hidden)

    action = torch.argmax(logits).item()

    if _last_action == action: ##dont repeat anti looping 
        _repeat += 1
    else:
        _repeat = 0

    if _repeat > 8:
        action = rng.integers(0, 5) ##random action to break the loop
        _repeat = 0

    _last_action = action

    return ACTIONS[action]