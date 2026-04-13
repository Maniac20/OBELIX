from __future__ import annotations
from typing import Optional, List
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]


class QNet(nn.Module):
    def __init__(self, obs_dim=18, hidden=128, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 3, hidden),   ##match with training state construction which concatenates current and two previous observations
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)


##gloabl model 
_hidden = None

_model: Optional[QNet] = None




def _load_once():
    global _model
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")

    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found next to agent.py")

    model = QNet()

    sd = torch.load(wpath, map_location="cpu")

    # handle wrapped checkpoints
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model.load_state_dict(sd, strict=True)
    model.eval()

    _model = model


_last_action = None
_repeat = 0
_prev_obs = None
@torch.no_grad()
def policy(obs, rng):
    global _hidden, _last_action, _repeat, _prev_obs

    _load_once()
    if _prev_obs is None:
        _prev_obs = [np.zeros_like(obs), np.zeros_like(obs)]

    state = np.concatenate([obs, _prev_obs[0], _prev_obs[1]]) ##build the state 
    x = torch.tensor(state, dtype=torch.float32)

    q_vals = _model(x) ##Q interference 
    action = torch.argmax(q_vals).item()

  
    if _last_action == action: ##this is anti loop logic, 
        # if we repeat the same action too many times, we force a random action to break out of potential loops
        _repeat += 1
    else:
        _repeat = 0

    if _repeat > 8:
        action = rng.integers(0, 5)
        _repeat = 0

    _last_action = action

    # update history
    _prev_obs = [obs.copy(), _prev_obs[0]]

    return ACTIONS[action]