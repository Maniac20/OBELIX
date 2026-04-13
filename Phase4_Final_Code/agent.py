from __future__ import annotations
from typing import Optional, List
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]


##keep this same as training ,otherwise the weights won't load
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

_model: Optional[DDQN_LSTM] = None
_hidden = None
_last_action = None
_repeat = 0



def _load_once():
    global _model
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")

    if not os.path.exists(wpath):
        raise FileNotFoundError("weights.pth not found")

    model = DDQN_LSTM()

    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model.load_state_dict(sd, strict=True)
    model.eval()

    _model = model



@torch.no_grad()
def policy(obs, rng):
    global _hidden, _last_action, _repeat

    _load_once()


    if obs[17] == 1:
        return rng.choice(["L45", "R45"])

    if _hidden is None:
        _hidden = (
            torch.zeros(1, 1, 128),
            torch.zeros(1, 1, 128)
        )

    # reset memory when no signal
    if np.sum(obs[:17]) == 0:
        _hidden = (
            torch.zeros(1, 1, 128),
            torch.zeros(1, 1, 128)
        )

    # detach hidden
    _hidden = (_hidden[0].detach(), _hidden[1].detach())

    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    q_vals, _hidden = _model(x, _hidden)
    action = torch.argmax(q_vals, dim=1).item()


    if _last_action == action:
        _repeat += 1
    else:
        _repeat = 0

    if _repeat > 8:
        action = rng.integers(0, 5)
        _repeat = 0

    _last_action = action

    return ACTIONS[action]