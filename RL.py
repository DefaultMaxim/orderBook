import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import pandas as pd
import matplotlib.pyplot as plt

def update_ow_state(conv_state: float, q_step: float, r: float) -> float:
    return conv_state * np.exp(-r) + q_step

class _BaseMultiOrderEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        num_agents=2, T=50,
        V0=100.0, sigma=0.01,
        s=0.02, r=0.1, lambd=1000.0,
        unfilled_weight=5.0,         # штраф за единицу неисполненного объёма в терминале
        seed=None
    ):
        super().__init__()
        self.num_agents = int(num_agents)
        self.T = int(T)
        self.V0 = float(V0)
        self.s = float(s)
        self.r = float(r)
        self.lambd = float(lambd)
        self.sigma = float(sigma)
        self.unfilled_weight = float(unfilled_weight)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_agents,), dtype=np.float32)

        obs_dim = 1 + self.num_agents + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # время и mid-траектория (брон. шум, dt=1)
        self.t = np.arange(self.T)
        noise = self._rng.normal(0.0, self.sigma, size=self.T)
        self.V = self.V0 + np.cumsum(noise)

        self.current_step = 0
        # у каждого агента фиксированный объём X_i=1
        self.remaining = np.ones(self.num_agents, dtype=np.float32)
        # история покадровых исполнений (для графиков)
        self.exec_hist = np.zeros((self.num_agents, self.T), dtype=np.float32)

        self.conv_state = 0.0

        return self._get_obs(), {}

    def _map_action(self, action):
        a = np.clip(action, -1.0, 1.0)
        return ((a + 1.0) / 2.0).astype(np.float32)

    def _get_obs(self):
        mid = self.V[min(self.current_step, self.T-1)]
        progress = self.current_step / self.T
        obs = np.concatenate([[mid], self.remaining, [progress]]).astype(np.float32)
        return obs

    def _price_and_update_state(self, q_step_total: float) -> float:
        raise NotImplementedError

    def step(self, action):
        frac = self._map_action(action)            # [0,1]
        executed = frac * self.remaining
        self.remaining -= executed
        self.exec_hist[:, self.current_step] = executed
        q_total = float(np.sum(executed))

        A_t = self._price_and_update_state(q_total)
        V_t = self.V[self.current_step]
        
        step_cost = q_total * (A_t - V_t)
        reward = float(-step_cost)

        # шаг времени
        self.current_step += 1
        terminated = bool(self.current_step >= self.T)
        truncated = False

        # в терминале штраф за недоисполнение
        if terminated:
            unfilled = float(np.sum(self.remaining))
            reward -= self.unfilled_weight * unfilled

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

class MultiOrderEnvNoImpact(_BaseMultiOrderEnv):
    def _price_and_update_state(self, q_step_total: float) -> float:
        return self.V[self.current_step] + self.s / 2.0
    
class MultiOrderEnvOW(_BaseMultiOrderEnv):
    def _price_and_update_state(self, q_step_total: float) -> float:
        # обновляем состояние импакта на текущем шаге
        self.conv_state = update_ow_state(self.conv_state, q_step_total, self.r)
        return self.V[self.current_step] + self.s / 2.0 + self.conv_state / self.lambd


def act_twap(env: _BaseMultiOrderEnv):
    """TWAP: равный объём на оставшиеся шаги (переводим в долю остатка)."""
    steps_left = max(env.T - env.current_step, 1)
    # желаемый объём на шаг по каждому агенту
    desired = env.remaining / steps_left
    frac = np.zeros_like(env.remaining)
    nz = env.remaining > 1e-8
    frac[nz] = (desired[nz] / env.remaining[nz]).astype(np.float32)
    # маппим долю [0,1] обратно в действие в пространстве [-1,1]
    action = 2.0 * np.clip(frac, 0.0, 1.0) - 1.0
    return action

def act_all_in(env: _BaseMultiOrderEnv):
    """ALL-IN: исполняем весь остаток в текущий шаг (доля=1)."""
    frac = np.ones_like(env.remaining, dtype=np.float32)
    action = 2.0 * frac - 1.0
    return action

def evaluate(env_class, label, train_steps=100_000, n_runs=5, seed=42):
    # обучаем PPO в данной среде
    train_env = env_class(seed=seed)
    model = PPO("MlpPolicy", train_env, verbose=0)
    model.learn(total_timesteps=train_steps)

    def _run_once(strategy):
        env = env_class(seed=seed+123)  # другой сид для теста
        obs, _ = env.reset()
        done = False
        total_cost = 0.0
        while not done:
            if strategy == "ppo":
                action, _ = model.predict(obs)
            elif strategy == "twap":
                action = act_twap(env)
            elif strategy == "all":
                action = act_all_in(env)
            else:
                raise ValueError("unknown strategy")
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            total_cost += -reward
        # агрегаты для графиков
        cum_exec = np.cumsum(env.exec_hist.sum(axis=0))
        exec_profile = env.exec_hist.sum(axis=0)
        V_traj = env.V.copy()
        return total_cost, cum_exec, exec_profile, V_traj

    out = {}
    curves = {}
    for strat in ["ppo", "twap", "all"]:
        costs = []
        last_cum = None; last_prof = None; last_V = None
        for _ in range(n_runs):
            c, cum, prof, Vtr = _run_once(strat)
            costs.append(c)
            last_cum, last_prof, last_V = cum, prof, Vtr
        out[f"{label}_{strat.upper()}"] = (float(np.mean(costs)), float(np.std(costs)))
        curves[strat] = (last_cum, last_prof, last_V)

    return out, curves


