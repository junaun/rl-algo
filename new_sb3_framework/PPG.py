from stable_baselines3 import PPO
import torch as th
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.categorical import Categorical
from torch import distributions as td
import torch.nn.functional as F

class PPG(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_aux_epochs = 6
        self.aux_batch_size = 8
        self.beta_clone = 1.0
    
    def _set_paper_parameters(self):
        if self.env.num_envs != 64:
            print("Warning: Paper uses 64 environments. "
                  "Change this if you want to have the same setup.")

        self.learning_rate = 5e-4
        self.aux_learning_rate = 5e-4
        self.n_steps = 256
        self.batch_size = 8
        self.aux_batch_size = 4
        self.n_policy_iters = 32
        self.n_epochs = 1
        self.n_aux_epochs = 6
        self.gamma = 0.999
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.clip_range_vf = None
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.beta_clone = 1.0
        self.vf_true_coef = 1.0
        self.max_grad_norm = 0.5
        self.target_kl = None
        self.policy_kwargs["activation_fn"] = th.nn.Identity
    
    def train(self) -> None:
        super().train() 
        aux_obs = self.rollout_buffer.observations
        aux_obs = th.as_tensor(aux_obs).to(self.policy.device)
        aux_return = self.rollout_buffer.returns
        aux_return = th.as_tensor(aux_return).to(self.policy.device)
    
        with th.no_grad():
            distribution, _, _ = self.policy.forward_policy(aux_obs)
            old_pd = distribution.distribution

        # Create TensorDataset and DataLoader Minibatch
        dataset = TensorDataset(aux_obs, aux_return, old_pd.logits)
        dataloader = DataLoader(dataset, batch_size=self.aux_batch_size, shuffle=False)

        # Update aux with custom epochs
        for aux_epoch in range(self.n_aux_epochs):
            for aux_obs, aux_return, old_pd in dataloader:
                # Get aux policy outputs
                new_distribution, new_value, new_aux_return = self.policy.forward_aux(aux_obs)
                new_pd = new_distribution.distribution

                old_pd = Categorical(logits=old_pd)
                # Calculate losses
                kl_loss = td.kl_divergence(old_pd, new_pd).mean()
                real_value_loss = 0.5 * F.mse_loss(new_value, aux_return)
                aux_value_loss = 0.5 * F.mse_loss(new_aux_return, aux_return)
                joint_loss = aux_value_loss + self.beta_clone * kl_loss

                # Update aux
                loss = (joint_loss + real_value_loss)
                self.policy.aux_optimizer.zero_grad()  
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # if (i + 1) % args.n_aux_grad_accum == 0:
                    # nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                self.policy.aux_optimizer.step()
                # Clip grad norm
                # th.nn.utils.clip_grad_norm_(self.policy.parameters(),
                #                             self.max_grad_norm)
    