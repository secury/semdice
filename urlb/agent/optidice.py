from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import torch.distributions as pyd
from agent.ddpg import Actor, Critic


EPS = 1e-6


class ValueNetwork(nn.Module):
    def __init__(self, obs_type, input_dim, feature_dim, hidden_dim, output_activation=None):
        super().__init__()

        self.obs_type = obs_type
        self.output_activation = output_activation

        if obs_type == 'pixels':
            raise NotImplementedError()
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim
        
        assert self.output_activation in [None, 'exp', 'softplus']
        self.head = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(utils.weight_init)

    def forward(self, *inputs):
        assert self.obs_type != 'pixels'
        inputs = torch.cat(inputs, dim=-1)
        h = self.trunk(inputs)
        value = self.head(h)

        if self.output_activation == 'exp':
            value = torch.exp(value)
        elif self.output_activation == 'softplus':
            value = F.softplus(value)

        return value  # [batch_size, 1]


def schisq_f(x):
    return torch.where(x < 1, x * torch.log(x + 1e-10) - x + 1, (0.5 * (x - 1) ** 2))

def schisq_f_prime_inv(x):
    return torch.where(x < 0, torch.exp(x.clamp_max(0)),  x + 1)

def schisq_log_relu_f_prime_inv(x):
    return torch.where(x < 0, x, torch.log(x.clamp_min(0) + 1))

def schisq_f_pos_conj(y):
    return torch.where(y < 0, torch.exp(y.clamp_max(0)) - 1, 0.5 * y ** 2 + y)

def schisq_f_pos_conj_prime(y):
    return torch.where(y < 0, torch.exp(y.clamp_max(0)), y + 1)


def chisq_f(x):
    return 0.5 * (x - 1) ** 2

def chisq_f_prime_inv(x):
    return x + 1

def chisq_log_relu_f_prime_inv(x):
    return torch.log((x + 1).clamp_min(1e-6))

def chisq_f_pos_conj(y):
    return 0.5 * torch.relu(y + 1) ** 2 - 0.5

def chisq_f_pos_conj_prime(y):
    return torch.relu(y + 1)


POLY_C = 1.5

def poly_f(x):
    return 0.5 * torch.abs(x - 1) ** POLY_C

def poly_f_prime_inv(x):
    return torch.where(
        x < 0,
        1 - (-2*x / POLY_C)**(1/(POLY_C-1)),
        1 + (+2*x / POLY_C)**(1/(POLY_C-1)),
    )

def poly_log_relu_f_prime_inv(x):
    return torch.where(
        x < 0,
        torch.log(torch.relu(1 - (-2*x / POLY_C)**(1/(POLY_C-1))).clamp_min(1e-6)),
        torch.log(torch.relu(1 + (+2*x / POLY_C)**(1/(POLY_C-1))).clamp_min(1e-6)),
    )

def poly_f_pos_conj(y):
    return torch.where(
        y < 0,
        (1 - (-2*y/POLY_C)**(1/(POLY_C-1))) * y + 0.5 * (-2*y/POLY_C)**(POLY_C/(POLY_C-1)),
        (1 + (+2*y/POLY_C)**(1/(POLY_C-1))) * y - 0.5 * (+2*y/POLY_C)**(POLY_C/(POLY_C-1)),
    )


class OptiDICEAgent:
    def __init__(self,
                 name,
                 reward_free,
                 obs_type,
                 obs_shape,
                 action_shape,
                 device,
                 lr,
                 feature_dim,
                 hidden_dim,
                 num_expl_steps,
                 update_every_steps,
                 stddev_schedule,
                 nstep,
                 batch_size,
                 stddev_clip,
                 init_critic,
                 use_tb,
                 use_wandb,
                 alpha,
                 f_type,
                 nu_penalty_lambda, 
                 nu_gp_threshold, 
                 actor_num_components: int,
                 f_divergence_target: float,
                 update_encoder=False,
                 ):
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.solved_meta = None

        if alpha > 0:
            self.train_alpha = False
            self.alpha = alpha
        else:
            self.train_alpha = True
            self.log_alpha = torch.tensor(0.0, requires_grad=True)
            self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
            self.f_divergence_target = f_divergence_target

        self.f_type = f_type
        self.actor_num_components = actor_num_components

        self.nu_gp_threshold = nu_gp_threshold
        self.nu_penalty_lambda = nu_penalty_lambda

        # models and optimizers
        assert obs_type != 'pixels'
        if obs_type == 'pixels':
            raise NotImplementedError()                                 
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0]
            self.encoder_opt = None

        self.nu = ValueNetwork(obs_type, self.obs_dim, feature_dim, hidden_dim).to(device)
        self.nu_opt = torch.optim.Adam(self.nu.parameters(), lr=lr)

        # self.q1 = ValueNetwork(obs_type, self.obs_dim + self.action_dim, feature_dim, hidden_dim).to(device)
        # self.q2 = ValueNetwork(obs_type, self.obs_dim + self.action_dim, feature_dim, hidden_dim).to(device)
        # self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.critic = Critic(obs_type, self.obs_dim, self.action_dim, feature_dim, hidden_dim).to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.actor = Actor(self.obs_type, self.obs_dim, self.action_dim, feature_dim, hidden_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        rms_sa = utils.RMS(self.device)
        self.pbe_sa = utils.PBE(rms_sa, knn_clip=0.0, knn_k=12, knn_avg=True, knn_rms=False, device=self.device, shift=1e-6)

        # f-divergence
        if f_type == 'chisq':
            self.f = chisq_f
            self.f_prime_inv = chisq_f_prime_inv
            self.log_relu_f_prime_inv = chisq_log_relu_f_prime_inv
            self.f_pos_conj = chisq_f_pos_conj
            self.f_pos_conj_prime = chisq_f_pos_conj_prime
        elif f_type == 'softchisq':
            self.f = schisq_f
            self.f_prime_inv = schisq_f_prime_inv
            self.log_relu_f_prime_inv = schisq_log_relu_f_prime_inv
            self.f_pos_conj = schisq_f_pos_conj
            self.f_pos_conj_prime = schisq_f_pos_conj_prime
        else:
            raise NotImplementedError()

        self.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.nu.train(training)
        self.critic.train(training)
        self.actor.train(training)

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

    def act(self, obs, meta, step, eval_mode):
        if len(obs.shape) == 1:
            single_sample = True
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        else:
            single_sample = False
            obs = torch.as_tensor(obs, device=self.device)
        h = self.encoder(obs)
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)

        dist = self.actor.forward(obs, std=0.2)
        action = dist.mean if eval_mode else dist.sample(clip=None)

        if not eval_mode and step < self.num_expl_steps:
            action.uniform_(-1.0, 1.0)

        return action.cpu().numpy()[0] if single_sample else action.cpu().numpy()

    def update_actor_critic(self, obs, action, reward, discount, next_obs, init_obs, step):
        metrics = dict()
        batch_size = obs.shape[0]
        init_obs = obs

        # alpha
        if self.train_alpha:
            alpha = torch.exp(self.log_alpha).detach()
        else:
            alpha = self.alpha
        metrics.update({'alpha': alpha})

        # feedforward nu
        nu_all = self.nu(torch.concat([obs, init_obs, next_obs], dim=0))
        nu, nu_initial, next_nu = torch.split(nu_all, [batch_size, batch_size, batch_size])

        # q-network training
        q1, q2 = self.critic(obs, action)
        with torch.no_grad():
            q_target = (reward + discount * next_nu).detach()
        critic_loss = ((q1 - q_target) ** 2).mean() + ((q2 - q_target) ** 2).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # nu-network training
        e_hat = reward + discount * next_nu - nu  # [batch_size, 1]

        # Calculates the weights of the samples using w
        w_analytic = F.relu(self.f_prime_inv(e_hat / alpha))

        nu_loss = (nu - discount * next_nu).mean() + alpha * self.f_pos_conj(e_hat / alpha).mean()  # full gradient

        #####
        # gradient penalty for nu
        if self.nu_penalty_lambda > 0:
            epsilon = torch.rand(batch_size, 1, device=self.device)
            obs1, obs2 = torch.concat([obs[:batch_size//2], init_obs[:batch_size//2]]), obs[torch.randperm(batch_size)]
            # obs1, obs2 = obs, obs_rand
            obs_inter = epsilon * obs1 + (1 - epsilon) * obs2
            obs_inter.requires_grad = True
            nu_inter = self.nu(obs_inter)
            grads_inter = torch.autograd.grad(outputs=nu_inter, inputs=obs_inter, grad_outputs=torch.ones_like(nu_inter), retain_graph=True, create_graph=True, only_inputs=True)[0]
            nu_grad_norm = torch.norm(grads_inter, dim=-1)  # [batch_size]
            nu_grad_penalty = (torch.maximum(torch.tensor(0), nu_grad_norm - self.nu_gp_threshold) ** 2).mean()
            nu_loss += self.nu_penalty_lambda * nu_grad_penalty
            metrics['nu_grad_norm'] = nu_grad_norm.mean().item()
            metrics['nu_grad_penalty'] = nu_grad_penalty.item()
        #####
        self.nu_opt.zero_grad(set_to_none=True)
        nu_loss.backward()
        self.nu_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['nu_loss'] = nu_loss.item()
            metrics['nu_mean'] = nu.mean().item()
            metrics['e_hat_mean'] = e_hat.mean().item()
            metrics['w_analytic_mean'] = w_analytic.mean().item()

        # Bellman flow constraint
        with torch.no_grad():
            flow_constraint_violation = (nu - discount * next_nu).mean() + (w_analytic * (discount * next_nu - nu)).mean()
            reward_w_analytic = (w_analytic * reward).mean()
            f_divergence_w_analytic = self.f(w_analytic).mean()

            w = F.relu(self.f_prime_inv((q1 - nu) / alpha))
            reward_w = (w * reward).mean()
            f_divergence_w = self.f(w).mean()
            metrics.update({
                'flow_constraint_violation': flow_constraint_violation,
                'reward_w_analytic': reward_w_analytic.item(),
                'f_divergence_w_analytic': f_divergence_w_analytic.item(),
                'w_mean': w.mean().item(),
                'reward_w': reward_w.item(),
                'f_divergence_w': f_divergence_w.item(),
            })
            metrics.update({
                'q_loss': critic_loss.item(), 'q1_mean': q1.mean().item()
            })

        # Actor training
        dist = self.actor(obs, std=0.2)
        action_policy = dist.sample(clip=0.3)
        log_policy_prob = dist.log_prob(action_policy).sum(-1, keepdim=True)
        q1_policy, q2_policy = self.critic(obs, action_policy)
        q_policy = torch.minimum(q1_policy, q2_policy)

        log_w_policy = self.log_relu_f_prime_inv((q_policy - nu.detach()) / alpha)
        # log_dD = -self.pbe_sa(torch.cat([obs, action_policy], dim=1), torch.cat([obs, action], dim=1))
        actor_loss = -log_w_policy.mean() #- alpha * log_dD.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # Train alpha
        if self.train_alpha:
            alpha_loss = -self.log_alpha * (self.f(w_analytic).mean().detach() - self.f_divergence_target)
            self.log_alpha_opt.zero_grad()
            alpha_loss.backward()
            self.log_alpha_opt.step()

        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_iter, step):
        metrics = dict()
        #import ipdb; ipdb.set_trace()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, init_obs = utils.to_torch(batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)
        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        # update
        metrics.update(self.update_actor_critic(obs, action, reward, discount, next_obs, init_obs, step))

        return metrics
