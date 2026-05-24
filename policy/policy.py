import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import gym
import torch
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

from model.model import HGAT


class HGATNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.
    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """
    def __init__(
            self,
            num_stocks: int,
            obs_cols: int,
            n_head=8,
            hidden_dim=128,
            no_ind=False,
            no_neg=False,
    ):
        super(HGATNetwork, self).__init__()
        # The HGAT generator outputs one score per stock, so the policy/value
        # latent dimension equals the number of stocks.
        self.num_stocks = num_stocks
        self.n_features = obs_cols - 3 * num_stocks  # raw per-stock features (d)
        self.latent_dim_pi = num_stocks
        self.latent_dim_vf = num_stocks

        self.policy_net = HGAT(num_stocks=num_stocks, n_features=self.n_features,
                               num_heads=n_head, hidden_dim=hidden_dim,
                               no_ind=no_ind, no_neg=no_neg)
        self.value_net = HGAT(num_stocks=num_stocks, n_features=self.n_features,
                              num_heads=n_head, hidden_dim=hidden_dim,
                              no_ind=no_ind, no_neg=no_neg)

    def _pack(self, features: torch.Tensor) -> torch.Tensor:
        # SB3 flattens the (N, d+3N) observation row-major. Decode it back and
        # re-pack into the [ind; pos; neg; feat^T] block layout HGAT expects.
        batch = features.shape[0]
        n, d = self.num_stocks, self.n_features
        x = features.view(batch, n, d + 3 * n)
        feat = x[:, :, :d]
        ind = x[:, :, d:d + n]
        pos = x[:, :, d + n:d + 2 * n]
        neg = x[:, :, d + 2 * n:d + 3 * n]
        packed = torch.cat([ind, pos, neg, feat.transpose(1, 2)], dim=1)
        return packed.reshape(batch, -1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            :return: (th.Tensor, th.Tensor) latent_policy, latent_value of     the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
            """
        packed = self._pack(features)
        return self.policy_net(packed), self.value_net(packed)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(self._pack(features))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(self._pack(features))

class HGATActorCriticPolicy(ActorCriticPolicy):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable[[float], float],
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 *args,
                 **kwargs,
                 ):
        super(HGATActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = HGATNetwork(num_stocks=self.observation_space.shape[0],
                                         obs_cols=self.observation_space.shape[1],
                                         )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        actions, values, log_prob = super().forward(obs, deterministic)
        return actions, values, log_prob

    def _predict(self, observation, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        actions, values, log_prob = self.forward(observation, deterministic)
        return actions
