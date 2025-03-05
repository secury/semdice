from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import torch.distributions as pyd


EPS = 1e-6


class Actor(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()
        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

        self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        policy_layers = []
        policy_layers += [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        # add additional hidden layer for pixels
        if obs_type == 'pixels':
            policy_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class TanhMixtureGaussianActor(nn.Module):

    def __init__(self, obs_dim, action_dim, num_components):
        super().__init__()
        feature_dim, hidden_dim = 1024, 1024
        self.mean_bound_min, self.mean_bound_max = [-7, 7]
        self.log_std_bound_min, self.log_std_bound_max = [-5, 3]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_components = num_components
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh(),
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True),
        )
        self.means = nn.Linear(hidden_dim, action_dim * num_components)
        self.log_stds = nn.Linear(hidden_dim, action_dim * num_components)
        self.logits = nn.Linear(hidden_dim, num_components)
        self.apply(utils.weight_init)

    def forward(self, obs):
        batch_size, _ = obs.shape
        h = self.trunk(obs)  # [batch_size, hidden_dim]
        means = self.means(h).view(batch_size, self.num_components, self.action_dim)
        means = self.mean_bound_min + 0.5 * (self.mean_bound_max - self.mean_bound_min) * (torch.tanh(means) + 1)
        log_stds = self.log_stds(h).view(batch_size, self.num_components, self.action_dim)
        log_stds = self.log_std_bound_min + 0.5 * (self.log_std_bound_max - self.log_std_bound_min) * (torch.tanh(log_stds) + 1)

        stds = log_stds.exp()
        component_logits = self.logits(h)  # [batch_size, num_components]

        return means, stds, component_logits

    def sample(self, obs, eval_mode=False):
        batch_size = obs.shape[0]
        means, stds, component_logits = self.forward(obs)

        pretanh_actions_dist = pyd.Normal(means, stds)
        if eval_mode:
            pretanh_actions = pretanh_actions_dist.mean  # [batch_size, num_components, action_dim]
        else:
            pretanh_actions = pretanh_actions_dist.sample()  # [batch_size, num_components, action_dim]

        component_dist = pyd.Categorical(logits=component_logits)

        component = component_dist.sample()  # [batch_size]

        pretanh_action = pretanh_actions[torch.arange(batch_size), component]  # [batch_size, action_dim]
        action = torch.tanh(pretanh_action)

        return action

    def log_prob(self, obs, action):
        means, stds, component_logits = self.forward(obs)
        pretanh_actions_dist = pyd.Normal(means, stds)  # [batch_size, num_components, action_dim]

        pretanh_action = torch.atanh(action.clamp(-1 + EPS, 1 - EPS))  # [batch_size, action_dim]
        component_log_prob = component_logits - torch.logsumexp(component_logits, dim=-1, keepdim=True)  # [batch_size, num_components]

        pretanh_actions = torch.tile(pretanh_action[:, None, :], (1, self.num_components, 1))  # [batch_size, num_components, action_dim]
        pretanh_log_prob = torch.logsumexp(
            component_log_prob + pretanh_actions_dist.log_prob(pretanh_actions).sum(dim=-1),
            dim=-1, keepdim=True)  # [batch_size, 1]

        log_prob = pretanh_log_prob - torch.log(1 - action ** 2 + EPS).sum(dim=-1, keepdim=True)

        info = {'means': means, 'stds': stds, 'component_logits': component_logits}

        return log_prob, info

    def sample_and_log_prob(self, obs):
        batch_size = obs.shape[0]
        means, stds, component_logits = self.forward(obs)

        pretanh_actions_dist = pyd.Normal(means, stds)  # [batch_size, num_components, action_dim]
        component_dist = pyd.Categorical(logits=component_logits)

        pretanh_actions_sample = pretanh_actions_dist.rsample()  # [batch_size, num_components, action_dim]
        # component_relaxed_onehot = F.gumbel_softmax(component_logits, tau=0.8, hard=True)  # [batch_size, num_components]
        # pretanh_action = (pretanh_actions_sample * component_relaxed_onehot[:, :, None]).sum(dim=1)  # [batch_size, action_dim]
        component = component_dist.sample()  # [batch_size]
        pretanh_action = pretanh_actions_sample[torch.arange(batch_size), component]  # [batch_size, action_dim]

        pretanh_actions = torch.tile(pretanh_action[:, None, :], (1, self.num_components, 1))  # [batch_size, num_components, action_dim]
        action = torch.tanh(pretanh_action)

        component_log_prob = component_logits - torch.logsumexp(component_logits, dim=-1, keepdim=True)  # [batch_size, num_components]

        pretanh_log_prob = torch.logsumexp(
            component_log_prob + pretanh_actions_dist.log_prob(pretanh_actions).sum(dim=-1),
            dim=-1, keepdim=True)  # [batch_size, 1]

        log_prob = pretanh_log_prob - torch.log(1 - action ** 2 + EPS).sum(dim=-1, keepdim=True)

        info = {
            'means': means,
            'stds': stds,
            'component_logits': component_logits,
        }

        return action, log_prob, info

    def sample_all_components(self, obs):
        means, stds, component_logits = self.forward(obs)
        pretanh_actions_dist = pyd.Normal(means, stds)  # [batch_size, num_components, action_dim]

        pretanh_all_actions = pretanh_actions_dist.rsample()  # [batch_size, num_components, action_dim]
        all_actions = torch.tanh(pretanh_all_actions)
        component_probs = F.softmax(component_logits, dim=-1)  # [batch_size, num_components]

        info = {
            'means': means,
            'stds': stds,
            'component_logits': component_logits,  # [batch_size, num_components]
            'component_probs': component_probs,  # [batch_size, num_components]
        }

        return all_actions, info


class MixtureTruncatedGaussianActor(nn.Module):
    """Use fixed std"""

    def __init__(self, obs_dim, action_dim, num_components, std, std_clip):
        super().__init__()
        feature_dim, hidden_dim = 1024, 1024
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_components = num_components
        self.std = std
        self.std_clip = std_clip

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh(),
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True),
        )
        self.means = nn.Linear(hidden_dim, action_dim * num_components)
        self.logits = nn.Linear(hidden_dim, num_components)
        self.apply(utils.weight_init)

    def forward(self, obs, std):
        batch_size, _ = obs.shape

        h = self.trunk(obs)  # [batch_size, hidden_dim]
        means = self.means(h).view(batch_size, self.num_components, self.action_dim)
        stds = torch.ones_like(means) * std
        component_logits = self.logits(h)  # [batch_size, num_components]

        actions_dist = utils.TruncatedNormal(torch.tanh(means), stds)
        component_dist = pyd.Categorical(logits=component_logits)

        return actions_dist, component_dist

    def sample(self, obs, eval_mode):
        batch_size = obs.shape[0]
        actions_dist, component_dist = self.forward(obs, std=0 if eval_mode else self.std)

        actions = actions_dist.sample(clip=self.std_clip)  # [batch_size, num_components, action_dim]
        component = component_dist.sample()  # [batch_size]

        action = actions[torch.arange(batch_size), component]  # [batch_size, action_dim]

        return action

    def log_prob(self, obs, action):
        actions_dist, component_dist = self.forward(obs, self.std)

        actions = torch.tile(action[:, None, :], (1, self.num_components, 1))  # [batch_size, num_components, action_dim]
        component_log_prob = component_dist.logits - torch.logsumexp(component_dist.logits, dim=-1, keepdim=True)  # [batch_size, num_components]
        log_prob = torch.logsumexp(
            component_log_prob + actions_dist.log_prob(actions).sum(dim=-1),
            dim=-1, keepdim=True)  # [batch_size, 1]

        return log_prob

    def sample_and_log_prob(self, obs):
        batch_size = obs.shape[0]
        actions_dist, component_dist = self.forward(obs, self.std)

        all_actions = actions_dist.sample(clip=self.std_clip)  # [batch_size, num_components, action_dim]
        component = component_dist.sample()  # [batch_size]
        action = all_actions[torch.arange(batch_size), component]  # [batch_size, action_dim]

        actions = torch.tile(action[:, None, :], (1, self.num_components, 1))  # [batch_size, num_components, action_dim]
        component_log_prob = component_dist.logits - torch.logsumexp(component_dist.logits, dim=-1, keepdim=True)  # [batch_size, num_components]
        log_prob = torch.logsumexp(
            component_log_prob + actions_dist.log_prob(actions).sum(dim=-1),
            dim=-1, keepdim=True)  # [batch_size, 1]

        info = {
            'means': actions_dist.mean,
            'stds': actions_dist.stddev,
            'component_logits': component_dist.logits,
        }

        return action, log_prob, info

    def sample_all_components(self, obs):
        actions_dist, component_dist = self.forward(obs, self.std)

        all_actions = actions_dist.sample()  # [batch_size, num_components, action_dim]
        component_probs = F.softmax(component_dist.logits, dim=-1)  # [batch_size, num_components]

        info = {
            'means': actions_dist.mean,
            'stds': actions_dist.stddev,
            'component_logits': component_dist.logits,  # [batch_size, num_components]
            'component_probs': component_probs,  # [batch_size, num_components]
        }

        return all_actions, info


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class Critic(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type
        hidden_depth = 2
        self.Q1 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2

class SACAgent():
    """SAC algorithm."""
    def __init__(self,
                 name,
                 obs_type,
                 obs_shape,
                 action_shape,
                 device,
                 lr,
                 feature_dim,
                 hidden_dim,
                 critic_target_tau,
                 num_expl_steps,
                 update_every_steps,
                 nstep,
                 batch_size,
                 init_critic,
                 use_tb,
                 use_wandb,
                 init_temperature,
                 critic_target_update_frequency,
                 mdn_num_components_actor,
                 learnable_temperature,
                 meta_dim=0, 
                 update_encoder=False):

        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.solved_meta = None
        self.init_temperature = init_temperature
        self.critic_target_update_frequency = critic_target_update_frequency
        self.mdn_num_components_actor = mdn_num_components_actor
        self.learnable_temperature = learnable_temperature

        assert obs_type != 'pixels'
        if obs_type == 'pixels':
            raise NotImplementedError()                                 
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim

        self.obs_dim = obs_shape[0] + meta_dim
        self.actor = TanhMixtureGaussianActor(self.obs_dim, self.action_dim, 
                                              num_components=self.mdn_num_components_actor).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)


        self.critic = Critic(obs_type, self.obs_dim, self.action_dim, 
                             feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(obs_type, self.obs_dim, self.action_dim, 
                             feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -self.action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta
    
    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)


    def act(self, obs, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = self.encoder(obs)
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        if eval_mode:
            action = self.actor.sample(inpt, eval_mode)
        else:
            action = self.actor.sample(inpt, eval_mode)
            action = (action + torch.randn_like(action) * 0.1).clamp(-1, 1)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]



    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()
        next_action, log_prob, _ = self.actor.sample_and_log_prob(next_obs)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        
        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = current_Q1.mean().item()
            metrics['critic_q2'] = current_Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return metrics

    def update_actor_and_alpha(self, obs, step):
        metrics = dict()
        action, log_prob, actor_info = self.actor.sample_and_log_prob(obs)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            if self.use_tb or self.use_wandb:
                metrics['alpha_loss'] = alpha_loss
                metrics['alpha_value'] = self.alpha
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics
        
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, init_obs = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()
            
        # update critic 
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor_and_alpha(obs.detach(), step))

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)
        
        return metrics