from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import torch.distributions as pyd
from agent.optidice import OptiDICEAgent, ValueNetwork
from agent.ddpg import Critic
from agent.icm_apt import ICM


EPS = 1e-6


class SEMDICEAgent(OptiDICEAgent):
    def __init__(self,
                 use_icm: bool,
                 icm_rep_dim: int,
                 p_init_data: bool,
                 mu_penalty_lambda,
                 mu_gp_threshold,
                 mu_rand_inter: bool,
                 **kwargs):
        super().__init__(**kwargs)

        self.use_icm = use_icm
        self.icm_rep_dim = icm_rep_dim
        self.p_init_data = p_init_data
        self.mu_penalty_lambda = mu_penalty_lambda
        self.mu_gp_threshold = mu_gp_threshold
        self.mu_rand_inter = mu_rand_inter

        self.mu = ValueNetwork(self.obs_type, self.obs_dim, self.feature_dim, self.hidden_dim).to(self.device)
        self.nu_mu_opt = torch.optim.Adam(list(self.nu.parameters()) + list(self.mu.parameters()), lr=self.lr)
        self.mu.train()

        self.obs_min = None
        self.obs_max = None

        if use_icm:
            self.icm = ICM(self.obs_dim, self.action_dim, self.hidden_dim, icm_rep_dim).to(self.device)
            self.icm_opt = torch.optim.Adam(self.icm.parameters(), lr=self.lr)
            self.icm.train()

            self.nu = ValueNetwork(self.obs_type, icm_rep_dim, self.feature_dim, self.hidden_dim).to(self.device)
            self.mu = ValueNetwork(self.obs_type, icm_rep_dim, self.feature_dim, self.hidden_dim).to(self.device)
            self.nu_mu_opt = torch.optim.Adam(list(self.nu.parameters()) + list(self.mu.parameters()), lr=self.lr)
            self.nu.train()
            self.mu.train()

        # particle-based entropy
        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(rms, knn_clip=0.0, knn_k=6, knn_avg=True, knn_rms=True, device=self.device, shift=1e-6)

        # for fine-tuning
        self.critic_target_tau = 0.01
        self.critic = Critic(self.obs_type, self.obs_dim, self.action_dim,
                             self.feature_dim, self.hidden_dim).to(self.device)
        self.critic_target = Critic(self.obs_type, self.obs_dim, self.action_dim,
                                    self.feature_dim, self.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.critic_target.train()

    def update_actor_critic(self, obs, action, extr_reward, discount, next_obs, init_obs, step, obs_min, obs_max):
        metrics = dict()
        batch_size = obs.shape[0]

        # alpha
        if self.train_alpha:
            alpha = torch.exp(self.log_alpha).detach()
        else:
            alpha = self.alpha
        metrics.update({'alpha': alpha})

        # nu-network training
        if self.use_icm:
            with torch.no_grad():
                obs_rep, init_obs_rep, next_obs_rep = self.icm.get_rep(obs), self.icm.get_rep(init_obs), self.icm.get_rep(next_obs)
                # rand_obs_rep = torch.rand_like(obs_rep) * 2 - 1  # [-1, 1]
                rand_obs_rep = self.icm.get_rep(torch.rand_like(obs) * (self.obs_max - self.obs_min) + self.obs_min)
        else:
            obs_rep, init_obs_rep, next_obs_rep = obs, init_obs, next_obs
            rand_obs_rep = torch.rand_like(obs) * (self.obs_max - self.obs_min) + self.obs_min

        if self.mu_rand_inter:
            epsilon = torch.rand(batch_size, 1, device=self.device)
            rand_obs_rep = epsilon * rand_obs_rep + (1 - epsilon) * obs_rep

        nu_all = self.nu(torch.concat([obs_rep, init_obs_rep, next_obs_rep], dim=0))
        nu, nu_initial, next_nu = torch.split(nu_all, [batch_size, batch_size, batch_size])
        mu_all = self.mu(torch.concat([obs_rep, init_obs_rep, rand_obs_rep], dim=0))
        mu, mu_initial, mu_p = torch.split(mu_all, [batch_size, batch_size, batch_size])

        e_hat = mu + discount * next_nu - nu
        w_analytic = F.relu(self.f_prime_inv(e_hat / alpha))

        # nu_mu_loss = ((1 - discount) * nu_initial).mean() \
        #     + alpha * self.f_pos_conj(e_hat / alpha).mean() \
        #     + torch.logsumexp(-mu_p + negative_log_p_s, dim=(0, 1))

        # no p0
        log_p_data_s = -self.pbe(obs) * self.obs_dim
        nu_mu_loss = (nu - discount * next_nu).mean() \
            + alpha * self.f_pos_conj(e_hat / alpha).mean() \
            + torch.logsumexp(-mu - log_p_data_s, dim=(0, 1))

        #####
        # gradient penalty for nu
        if self.nu_penalty_lambda > 0:
            epsilon = torch.rand(batch_size, 1, device=self.device)
            obs1, obs2 = torch.concat([obs_rep[:batch_size//2], init_obs_rep[:batch_size//2]]), obs_rep[torch.randperm(batch_size)]
            # obs1, obs2 = obs, obs_rand
            obs_inter = epsilon * obs1 + (1 - epsilon) * obs2
            obs_inter.requires_grad = True
            nu_inter = self.nu(obs_inter)
            grads_inter = torch.autograd.grad(outputs=nu_inter, inputs=obs_inter, grad_outputs=torch.ones_like(nu_inter), retain_graph=True, create_graph=True, only_inputs=True)[0]
            nu_grad_norm = torch.norm(grads_inter, dim=-1)  # [batch_size]
            nu_grad_penalty = (torch.maximum(torch.tensor(0), nu_grad_norm - self.nu_gp_threshold) ** 2).mean()
            nu_mu_loss += self.nu_penalty_lambda * nu_grad_penalty
            metrics['nu_grad_norm'] = nu_grad_norm.mean().item()
            metrics['nu_grad_penalty'] = nu_grad_penalty.item()
        #####
        #####
        # gradient penalty for mu
        if self.mu_penalty_lambda > 0:
            epsilon = torch.rand(batch_size, 1, device=self.device)
            # obs1, obs2 = obs[torch.randperm(batch_size)], obs[torch.randperm(batch_size)]
            obs1, obs2 = torch.concat([obs_rep[:batch_size//2], init_obs_rep[:batch_size//2]]), rand_obs_rep
            obs_inter = epsilon * obs1 + (1 - epsilon) * obs2
            obs_inter.requires_grad = True
            mu_inter = self.mu(obs_inter)
            grads_inter = torch.autograd.grad(outputs=mu_inter, inputs=obs_inter, grad_outputs=torch.ones_like(mu_inter), retain_graph=True, create_graph=True, only_inputs=True)[0]
            mu_grad_norm = torch.norm(grads_inter, dim=-1)  # [batch_size]
            mu_grad_penalty = (torch.maximum(torch.tensor(0), mu_grad_norm - self.mu_gp_threshold) ** 2).mean()
            nu_mu_loss += self.mu_penalty_lambda * mu_grad_penalty
            metrics['mu_grad_norm'] = mu_grad_norm.mean().item()
            metrics['mu_grad_penalty'] = mu_grad_penalty.item()
        #####

        self.nu_mu_opt.zero_grad(set_to_none=True)
        nu_mu_loss.backward()
        self.nu_mu_opt.step()

        # q-network training
        q1, q2 = self.critic(obs, action)
        with torch.no_grad():
            q_target = (mu + discount * next_nu).detach()
        critic_loss = ((q1 - q_target) ** 2).mean() + ((q2 - q_target) ** 2).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # Actor training
        dist = self.actor(obs, std=0.2)
        action_policy = dist.sample(clip=0.3)
        log_policy_prob = dist.log_prob(action_policy).sum(-1, keepdim=True)
        q1_policy, q2_policy = self.critic(obs, action_policy)
        q_policy = torch.minimum(q1_policy, q2_policy)
        # actor_loss = -q_policy.mean() + self.ent_coef * log_policy_prob.mean()
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

    def update_actor_critic_reward(self, obs, action, reward, discount, next_obs, step):
        """Fine-tuning"""
        metrics = dict()

        # Update critic
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        # Update actor
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()
        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics

    def update_icm(self, obs, action, next_obs, step):
        metrics = dict()
        forward_error, backward_error = self.icm(obs, action, next_obs)
        loss = forward_error.mean() + backward_error.mean()

        self.icm_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.icm_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['icm_loss'] = loss.item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, init_obs = utils.to_torch(batch, self.device)
        if self.p_init_data:
            init_obs = obs

        obs_min, obs_max = obs.min(dim=0)[0], obs.max(dim=0)[0]
        if self.obs_min is None or self.obs_max is None:
            self.obs_min, self.obs_max = obs_min, obs_max
        else:
            self.obs_min = torch.minimum(self.obs_min, obs_min)
            self.obs_max = torch.maximum(self.obs_max, obs_max)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        # update
        if self.reward_free:
            if self.use_icm:
                metrics.update(self.update_icm(obs, action, next_obs, step))

            metrics.update(self.update_actor_critic(
                obs, action, reward, discount, next_obs, init_obs, step, self.obs_min, self.obs_max))
        else:
            metrics.update(self.update_actor_critic_reward(
                obs, action, reward, discount, next_obs, step))

        return metrics

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)
