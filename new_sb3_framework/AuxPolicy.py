from typing import Any, Dict, Tuple
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
from functools import partial

class AuxActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        aux_lr_schedule = 5e-4
        super(AuxActorCriticPolicy, self).__init__(*args, **kwargs)
        lr_schedule = kwargs.get('lr_schedule', args[2])
        self.share_features_extractor = False
        self._build(lr_schedule, aux_lr_schedule)

    # Paper init
    @staticmethod
    def init_weights(module: th.nn.Module, gain: float = 1,
                     scale = 0.1) -> None:
        """
        Initialization with normalized fan-in (used in PPG paper)
        """
        if isinstance(module, (th.nn.Linear, th.nn.Conv2d)):
            module.weight.data *= scale / module.weight.norm(dim=1, p=2,
                                                             keepdim=True)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def _build(self, lr_schedule, aux_lr_schedule=None):
        if aux_lr_schedule is None:
            return

        super(AuxActorCriticPolicy, self)._build(lr_schedule)
        self.aux_head = th.nn.Linear(self.mlp_extractor.latent_dim_pi, 1)
        # self.aux_head.apply(lambda x: self.init_weights(x, gain=1))

        module_gains = {
            self.features_extractor: np.sqrt(2),
            self.mlp_extractor: np.sqrt(2),
            self.action_net: 0.01,
            self.value_net: 1,
            self.aux_head: 1
        }

        for module, gain in module_gains.items():
            with th.no_grad():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.aux_optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1),
            **self.optimizer_kwargs)

    def forward_policy(self, obs: th.Tensor
                       ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in the actor network

        :param obs: Observation
        :return: action, latent policy vector and latent value vector
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        distribution = self._get_action_dist_from_latent(latent_pi)
        return distribution, latent_pi, latent_vf

    def forward_aux(self, obs: th.Tensor
                    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks, including the auxiliary value network

        :param obs: Observation
        :return: action, true value and auxiliary value
        """
        distribution, latent_pi, latent_vf = self.forward_policy(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf.detach())
        aux_values = self.aux_head(latent_pi)
        return distribution, values, aux_values

    def forward(self, obs: th.Tensor, deterministic: bool = False,
                ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in the actor and critic networks

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        distribution, _, latent_vf = self.forward_policy(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super(AuxActorCriticPolicy, self)._get_constructor_parameters()

        data.update(
            dict(
                # dummy lr schedule, not needed for loading policy alone
                aux_lr_schedule=self._dummy_schedule,
            )
        )
        return data