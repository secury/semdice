import numpy as np
import random
from typing import Tuple, List
import logging
import scipy.special
from tqdm import tqdm
import pickle
import cvxpy as cp
import time
import scipy
import os
import copy
import argparse
import jax
import jax.numpy as jnp


np.set_printoptions(precision=3, suppress=True, linewidth=200)


class MDP:
    """MDP class."""

    def __init__(self,
                num_states: int,
                num_actions: int,
                transition: np.ndarray,
                reward: np.ndarray,
                gamma: float,
                p0: np.ndarray):
        """MDP Constructor.
        Args:
            num_states: the number of states.
            num_actions: the number of actions.
            transition: transition matrix. [num_states, num_actions, num_states].
            reward: reward function. [num_states, num_actions]
            gamma: discount factor (0 ~ 1).
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.transition = np.array(transition)
        self.reward = np.array(reward)
        self.gamma = gamma
        self.p0 = np.array(p0)
        assert self.transition.shape == (num_states, num_actions, num_states)
        assert self.reward.shape == (num_states, num_actions)
        assert self.p0.shape == (num_states,)

    def __copy__(self):
        mdp = MDP(
            num_states=self.num_states,
            num_actions=self.num_actions,
            transition=np.array(self.transition),
            reward=np.array(self.reward),
            gamma=self.gamma,
            p0=self.p0)
        return mdp


def policy_evaluation_mdp(mdp: MDP, pi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Policy evaluation (normalized value) for pi in the given MDP.
    Args:
        mdp: MDP instance.
        pi: a stochastic policy. [num_states, num_actions]
    Returns:
        (V_R, Q_R)
    """
    r = np.sum(mdp.reward * pi, axis=-1)
    p = np.sum(pi[:, :, None] * mdp.transition, axis=1)
    v = np.linalg.inv(np.eye(mdp.num_states) - mdp.gamma * p).dot(r)
    q = mdp.reward + mdp.gamma * mdp.transition.dot(v)
    return v, q


def generate_random_mdp(num_states: int, num_actions: int, gamma: float, dirichlet_param: float = 0.1):
    """Create a random CMDP.
    Args:
        num_states: the number of states.
        num_actions: the number of actions.
        gamma: discount factor (0 ~ 1).
    Returns:
        a MDP instance.
    """
    # Define a random transition.
    transition = np.zeros((num_states, num_actions, num_states))
    for s in range(num_states):
        for a in range(num_actions):
            # Transition to next states is defined sparsely.
            # For each (s,a), the connectivity to the next states is K.
            if a == 0 and False:
                transition[s, a, s] = 1
            else:
                K = 4
                p = np.r_[np.random.dirichlet(np.ones(K) * dirichlet_param), [0] * (num_states - K)]
                np.random.shuffle(p)
                transition[s, a, :] = p
            assert np.isclose(np.sum(transition[s, a, :]), 1)
    reward = np.random.rand(num_states, num_actions)
    p0 = np.eye(num_states)[0]
    # p0 = np.ones(num_states) / num_states
    mdp = MDP(num_states, num_actions, transition, reward, gamma, p0)

    return mdp


def generate_trajectory(seed: int,
                        mdp: MDP,
                        pi: np.ndarray,
                        num_episodes: int = 10,
                        max_timesteps: int = 50,
                        return_count: bool = False):
    """Generate trajectories using the policy in the CMDP.
    Args:
        seed: random seed.
        mdp: MDP instance.
        pi: a stochastic policy. [num_states, num_actions]
        num_episodes: the number of episodes to generate.
        max_timesteps: the maximum timestep within an episode.
    Returns:
        trajectory: list of list of (episode_idx, t, s_t, a_t, r_t, c_t, s_t').
    """
    if seed is not None:
        np.random.seed(seed + 1)

    def random_choice_prob_vectorized(p):
        """Batch random_choice.
        e.g. p = np.array([
                [0.1, 0.5, 0.4],
                [0.8, 0.1, 0.1]])
        Args:
            p: batch of probability vector.
        Returns:
            Sampled indices
        """
        r = np.expand_dims(np.random.rand(p.shape[0]), axis=1)
        return (p.cumsum(axis=1) > r).argmax(axis=1)

    N_sas1 = np.zeros((mdp.num_states, mdp.num_actions, mdp.num_states))
    N_sas1_discounted = np.zeros((mdp.num_states, mdp.num_actions, mdp.num_states))

    trajectory = [[] for i in range(num_episodes)]
    state = random_choice_prob_vectorized(np.tile(mdp.p0, (num_episodes, 1)))
    for t in range(max_timesteps):
        action = random_choice_prob_vectorized(p=pi[state, :])
        reward = mdp.reward[state, action]
        state1 = random_choice_prob_vectorized(p=mdp.transition[state, action, :])
        for i in range(num_episodes):
            trajectory[i].append(
                (i, t, state[i], action[i], reward[i], state1[i]))
            N_sas1[state[i], action[i], state1[i]] += 1
            N_sas1_discounted[state[i], action[i], state1[i]] += mdp.gamma ** t

        state = state1

    if return_count:
        return N_sas1, N_sas1_discounted
    else:
        return trajectory


def compute_mle_mdp(num_states: int,
                    num_actions: int,
                    reward: np.ndarray,
                    gamma: float,
                    trajectory) -> Tuple[MDP, np.ndarray]:
    """Construct a maximum-likelihood estimation MDP from the trajectories.
    Args:
        num_states: the number of states.
        num_actions: the number of actions.
        reward: reward function.
        gamma: discount factor (0~1).
        trajectory: trajectories collected by a behavior policy.
            list of list of (episode_idx, t, s_t, a_t, r_t, s_t').
    Returns:
        (MLE MDP, visitation count matrix)
    """
    n = np.zeros((num_states, num_actions, num_states))
    n0 = np.zeros(num_states)
    for trajectory_one in trajectory:
        # episode, t, s, a, r, s1
        for i, t, s, a, _, s1 in trajectory_one:
            n[s, a, s1] += 1
            if t == 0:
                n0[s] += 1
    transition = np.zeros((num_states, num_actions, num_states))
    for s in range(num_states):
        for a in range(num_actions):
            if n[s, a, :].sum() == 0:
                transition[s, a, :] = 1. / num_states
            else:
                transition[s, a, :] = n[s, a, :] / n[s, a, :].sum()

    p0 = n0 / n0.sum()
    mle_mdp = MDP(num_states, num_actions, transition, reward, gamma, p0)

    return mle_mdp, n


def compute_cb_exploration_policy(mdp: MDP, params, p_s, N_sas1: np.ndarray, is_state_only: bool, lr=0.1, niters=1):
    """
    Finds the exploration policy based on the state-action visitation counts.
    Compute a reward matrix based on inverse state-action visitation count, and solve for the optimimal policy

    Returns:
        policy
    """
    pi = scipy.special.softmax(params, axis=1)
    mdp = copy.copy(mdp)
    N_sa = N_sas1.sum(axis=-1)  # [num_states, num_actions]
    N_s = N_sa.sum(axis=-1, keepdims=True)  # [num_states, 1]

    if is_state_only:
        reward_exploration = np.power(N_s + 1e-4, -0.5) * np.ones((num_states, num_actions))
    else:
        reward_exploration = np.power(N_sa + 1e-4, -0.5)
    mdp.reward = reward_exploration
    mdp.reward = (mdp.reward - mdp.reward.min()) / (mdp.reward.max() - mdp.reward.min())

    # Perform policy gradients ascent
    for i in range(niters):
        V, Q = policy_evaluation_mdp(mdp, pi)
        params = params + lr * (p_s[:, None] * Q)
        pi = scipy.special.softmax(params, axis=1)
    params = np.log(pi + 1e-100)
    return pi, params


def compute_pb_entropy_policy(mdp: MDP, params: np.ndarray, p_s: np.ndarray, lr=0.1, niters=1):
    """
    Args:
        - mdp: MDP
        - pi: policy
        - d_s_data: [num_states]
    
    Return:
        a policy
    """
    pi = scipy.special.softmax(params, axis=1)
    mdp = copy.copy(mdp)
    mdp.reward = -np.log(p_s + 1e-100)[:, None] + np.zeros((mdp.num_states, mdp.num_actions))
    mdp.reward = (mdp.reward - mdp.reward.min()) / (mdp.reward.max() - mdp.reward.min())

    # Perform policy gradients ascent
    for i in range(niters):
        V, Q = policy_evaluation_mdp(mdp, pi)
        params = params + lr * (p_s[:, None] * Q)
        pi = scipy.special.softmax(params, axis=1)
    params = np.log(pi + 1e-100)
    return pi, params


def compute_stationary_distribution(mdp: MDP, pi: np.ndarray):
    """
    Compute marginal distribution for the given policy pi, d^pi(s,a).

    Return:
        - d^pi(s,a): [num_states, num_actions]
    """
    p0_s = mdp.p0
    p0_pi = (p0_s[:, None] * pi).reshape(mdp.num_states * mdp.num_actions)
    P_pi = (mdp.transition.reshape(mdp.num_states * mdp.num_actions,
                                   mdp.num_states)[:, :, None] * pi).reshape(
                                       mdp.num_states * mdp.num_actions,
                                       mdp.num_states * mdp.num_actions)
    r = mdp.reward.reshape(mdp.num_states * mdp.num_actions)
    if mdp.gamma < 1:
        d_pi = np.linalg.solve(np.eye(mdp.num_states * mdp.num_actions) - mdp.gamma * P_pi.T, (1 - mdp.gamma) * p0_pi)
        d_pi = d_pi.reshape(mdp.num_states, mdp.num_actions)
    else:
        # Define optimization variables
        d = cp.Variable(mdp.num_states * mdp.num_actions, pos=True)
        # Define the optimization problem
        f = d @ r
        constraints = [d == (1 - mdp.gamma) * p0_pi + mdp.gamma * P_pi.T @ d]
        if mdp.gamma == 1:
            constraints += [cp.sum(d) == 1]
        prob = cp.Problem(
            objective=cp.Maximize(f),
            constraints=constraints,
        )
        prob.solve(solver='MOSEK', verbose=False)

        d_pi = d.value.reshape(mdp.num_states, mdp.num_actions)
    return d_pi


@jax.jit
def f_semdice_chisq(x: jnp.ndarray, d_b, alpha, P, B):
    nu, mu = jnp.split(x, 2)
    e = B @ mu + mdp.gamma * P @ nu - (B @ nu) # [SA]
    f1 = (1 - mdp.gamma) * (mdp.p0 @ nu)
    f2 = alpha * d_b @ (0.5 * jnp.maximum(0, e / alpha + 1) ** 2 - 0.5)
    f3 = jax.scipy.special.logsumexp(-mu)
    return f1 + f2 + f3

f_semdice_chisq_grad = jax.jit(jax.grad(f_semdice_chisq))

def semdice_dual_gradient(mdp, params, d_b, alpha, lr=0.01, niters=1):
    """
    Args:
        - mdp: a MDP instance.
        - params: nu,mu
        - d_b: state-action stationary distribution of data-collection policy
        - alpha: float
    Return:
        - policy computed by solving the optimization problem using SEMDICE
    """
    B = np.repeat(np.eye(mdp.num_states), repeats=mdp.num_actions, axis=0)
    P = mdp.transition.reshape(mdp.num_states * mdp.num_actions, mdp.num_states)
    P = P / np.sum(P, axis=1, keepdims=True)
    d_b = d_b.reshape(mdp.num_states * mdp.num_actions)
    d_b_s = d_b.reshape(mdp.num_states, mdp.num_actions).sum(axis=1)

    for i in range(niters):
        params = params - lr * f_semdice_chisq_grad(params, d_b, alpha, P, B)
    nu, mu = np.split(np.asarray(params), 2)

    e = B @ mu + mdp.gamma * P @ nu - B @ nu
    w = np.maximum(e / alpha + 1, 0) + 1e-10
    d = (w * d_b).reshape(mdp.num_states, mdp.num_actions)
    d = d / np.sum(d)
    pi = d / np.sum(d, axis=1, keepdims=True)

    return pi, params


def compute_opitmal_maxent(mdp: MDP):
    B = np.repeat(np.eye(mdp.num_states), mdp.num_actions, axis=0)
    P = mdp.transition.reshape(mdp.num_states * mdp.num_actions, mdp.num_states)
    P = P / np.sum(P, axis=1, keepdims=True)

    # Define optimization variables
    d_bar = cp.Variable(mdp.num_states, pos=False)
    d = cp.Variable(mdp.num_states * mdp.num_actions, pos=True)

    # Define the optimization problem
    f = cp.sum(cp.entr(d_bar))

    constraints = []
    if mdp.gamma == 1:
        constraints += [(B.T - P.T) @ d == 0]
        constraints += [cp.sum(d) == 1]
    else:
        constraints += [(B.T - mdp.gamma * P.T) @ d == (1 - mdp.gamma) * mdp.p0]
    constraints += [B.T @ d == d_bar]
    prob = cp.Problem(
        objective=cp.Maximize(f),
        constraints=constraints,
    )
    result = prob.solve(solver='MOSEK', verbose=False)

    # Extract a policy
    d_bar, d = d_bar.value, d.value.reshape(mdp.num_states, mdp.num_actions)

    assert np.isclose(np.sum(d_bar), 1, atol=1e-4), (d_bar, np.sum(d_bar))
    assert np.isclose(np.sum(d), 1, atol=1e-4), (d, np.sum(d))
    assert np.all(d_bar >= 0), d_bar
    assert np.all(d >= 0), d

    pi = d + 1e-10
    pi = pi / np.sum(pi, axis=1, keepdims=True)
    return pi, {'d': d, 'd_bar': d_bar, 'f': result}


def compute_entropy(dist: np.ndarray):
    return -np.sum(dist * np.log(dist + 1e-10))


def run_experiment(
        mdp: MDP,
        num_max_trajectories: int,
        max_timesteps: int,
        method: str,
        method_params: dict,
        seed=0):
    # 1. Initialize parameters
    if method == 'semdice':
        params = np.zeros(mdp.num_states * 2)
    else:
        params = np.zeros((mdp.num_states, mdp.num_actions))

    ## Computes the state entropy of policy with uniform/MLE/count based exploration data collection
    pi_unif = np.ones((mdp.num_states, mdp.num_actions)) / mdp.num_actions
    unif_d_sa = compute_stationary_distribution(mdp, pi_unif)
    unif_d_s = unif_d_sa.sum(axis=-1)
    unif_d_s_entropy = compute_entropy(unif_d_s)

    pi_opt, _ = compute_opitmal_maxent(mdp)
    opt_d_sa = compute_stationary_distribution(mdp, pi_opt)  # [num_states, num_actions]
    opt_d_s = opt_d_sa.sum(axis=-1)
    opt_d_s_entropy = compute_entropy(opt_d_s)

    LOG_INDICES = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000, 100000]
    num_episodes = 1
    lr = 0.01
    semdice_alpha = method_params['semdice_alpha']

    pretrain_result = []
    pi = pi_unif

    mle_mdp = copy.copy(mdp)
    N_sas1 = np.zeros((mdp.num_states, mdp.num_actions, mdp.num_states)) + 1e-10
    N_sas1_gamma = np.zeros((mdp.num_states, mdp.num_actions, mdp.num_states)) + 1e-10
    pbar = tqdm(total=num_max_trajectories + 3, desc=f'{method}_{seed}', ncols=100)
    trajectory_idx = 0
    while trajectory_idx < num_max_trajectories + 3:
        epsilon = 1 / (trajectory_idx + 1)
        pi_eps_soft = epsilon * pi_unif + (1 - epsilon) * pi

        N_sas1_diff, N_sas1_gamma_diff = generate_trajectory(seed * 1000 + trajectory_idx, mdp, pi=pi_eps_soft, num_episodes=num_episodes, max_timesteps=max_timesteps, return_count=True)
        N_sas1 += N_sas1_diff
        N_sas1_gamma += N_sas1_gamma_diff

        trajectory_idx += num_episodes
        pbar.update(num_episodes)

        mle_mdp.transition = N_sas1 / np.sum(N_sas1, axis=-1, keepdims=True)

        data_d_sa = N_sas1_gamma.sum(axis=-1)  # [num_states, num_actions]
        data_d_sa = data_d_sa / np.sum(data_d_sa)
        data_d_s = data_d_sa.sum(axis=-1)  # [num_states]

        p_s = data_d_s

        if method == 'uniform':
            pi = pi_unif
        elif method == 'semdice':
            d_b = compute_stationary_distribution(mle_mdp, pi)
            pi, params = semdice_dual_gradient(mle_mdp, params, d_b, alpha=semdice_alpha, lr=lr, niters=100)
        elif method == 'cb_sa':
            pi, params = compute_cb_exploration_policy(mle_mdp, params, p_s, N_sas1, is_state_only=False, lr=lr, niters=100)
        elif method == 'cb_s':
            pi, params = compute_cb_exploration_policy(mle_mdp, params, p_s, N_sas1, is_state_only=True, lr=lr, niters=100)
        elif method == 'pb_s':
            pi, params = compute_pb_entropy_policy(mle_mdp, params, p_s, lr=lr, niters=100)
        else:
            raise NotImplementedError(method)

        if len(LOG_INDICES) > 0 and trajectory_idx >= LOG_INDICES[0]:
            LOG_INDICES.pop(0)

            pi_d_sa = compute_stationary_distribution(mdp, pi)
            pi_d_s = pi_d_sa.sum(axis=-1)
            pi_d_s_entropy = compute_entropy(pi_d_s)
            pi_d_s_entropy_normalized = (pi_d_s_entropy - unif_d_s_entropy) / (opt_d_s_entropy - unif_d_s_entropy)

            data_d_s_entropy = compute_entropy(data_d_s)
            data_d_s_entorpy_normalized = (data_d_s_entropy - unif_d_s_entropy) / (opt_d_s_entropy - unif_d_s_entropy)
            p_s_entropy = compute_entropy(p_s)

            # Log
            pretrain_result.append({
                'num_trajectories': trajectory_idx,
                'opt_d_s': opt_d_s,
                'opt_d_s_entropy': opt_d_s_entropy,
                'data_d_s': data_d_s,
                'data_d_s_entropy': data_d_s_entropy,
                'data_d_s_entorpy_normalized': data_d_s_entorpy_normalized,
                'unif_d_s': unif_d_s,
                'unif_d_s_entropy': unif_d_s_entropy,
                'pi_d_s': pi_d_s,
                'pi_d_s_entropy': pi_d_s_entropy,
                'pi_d_s_entropy_normalized': pi_d_s_entropy_normalized,
                'p_s': p_s,
                'p_s_entropy': p_s_entropy,
                'pi': pi,
            })
            print(
                f'\n'
                f'EXP_{method}_{seed}_{trajectory_idx}: ent_pi={pi_d_s_entropy:.6f}, ent_D={data_d_s_entropy:.6f} / '
                f'sub_pi={opt_d_s_entropy - pi_d_s_entropy:.6f}, sub_D={opt_d_s_entropy - data_d_s_entropy:.6f} / '
                f'pi_d_s_entropy_norm={pi_d_s_entropy_normalized:.6f} / '
                f'data_d_s_entorpy_normalized={data_d_s_entorpy_normalized:.6f} / '
                f'\n'
            )

    return pretrain_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_seed', default=0, type=int)
    parser.add_argument('--end_seed', default=10, type=int)
    parser.add_argument('--method', default='semdice', type=str)
    args = parser.parse_args()

    methods = [
        'semdice',
        'pb_s',
        'cb_s',
        'cb_sa',
        'uniform',
    ]
    assert args.method in methods, f"Undefined method: {args.method}"

    dirichlet_param = 1.0
    method = args.method
    num_states, num_actions, gamma = 20, 4, 0.95
    num_max_trajectories = 10000
    max_timesteps = 100

    os.makedirs('result_tabular', exist_ok=True)
    result_filepath = (
        f'result_tabular/{args.method}.pickle'
    )
    method_params = {
        'semdice_alpha': 0.1,
    }
    print('result_filepath: ', result_filepath)
    if os.path.exists(result_filepath):
        with open(result_filepath, 'rb') as f:
            result_all = pickle.load(f)
    else:
        result_all = {'pretrain': {}}  # seed -> result

    start_time = time.time()
    for seed in range(args.start_seed, args.end_seed):
        np.random.seed(seed * 10000)
        mdp = generate_random_mdp(num_states, num_actions, gamma, dirichlet_param)
        method = args.method
        if seed in result_all['pretrain']:
            print(f'SKIP ({seed}, {method})')
            continue
        pretrain_result = run_experiment(
            mdp,
            num_max_trajectories=num_max_trajectories,
            max_timesteps=max_timesteps,
            method=method,
            method_params=method_params,
            seed=seed)
        if os.path.exists(result_filepath):
            with open(result_filepath, 'rb') as f:
                result_all = pickle.load(f)
        result_all['pretrain'][seed] = pretrain_result

        print(f"{method:20s}: "
              f"{pretrain_result[-1]['opt_d_s_entropy'] - pretrain_result[-1]['pi_d_s_entropy']} "
              f"{pretrain_result[-1]['opt_d_s_entropy'] - pretrain_result[-1]['data_d_s_entropy']}")

        with open(f'{result_filepath}.{seed}.tmp', 'wb') as f:
            pickle.dump(result_all, f)
        os.rename(f'{result_filepath}.{seed}.tmp', result_filepath)
        print(f'Elapsed_time: {time.time() - start_time}')
        time.sleep(np.random.rand() * 0.1)
