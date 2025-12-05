import os
import math
import time
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions.normal import Normal

# Project imports
from tinyphysics import (
    TinyPhysicsModel,
    CONTROL_START_IDX,
    STEER_RANGE,
    CONTEXT_LENGTH,
    DEL_T,
    FUTURE_PLAN_STEPS,
    MAX_ACC_DELTA,
    ACC_G,
)

# --------------------
# Config
# --------------------
@dataclass
class Args:
    seed: int = 1
    cuda: bool = True
    torch_deterministic: bool = True

    # Data/sim config
    data_dir: str = "./data/test"
    model_path: str = "./models/tinyphysics.onnx"
    num_envs: int = 8
    num_steps: int = 256

    # Algo
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    # runtime (computed)
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# --------------------
# Utilities
# --------------------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# --------------------
# CleanRL-style Agent
# --------------------
class Agent(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 64, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1, device=device), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 64, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_act, device=device), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None):
        mean = self.actor_mean(obs)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        probs = Normal(mean, std)
        if action is None:
            action = mean + std * torch.randn_like(mean)
        # Sum over action dimensions
        logprob = probs.log_prob(action).sum(-1)
        entropy = probs.entropy().sum(-1)
        value = self.critic(obs)
        return action, logprob, entropy, value


def _resolve_steer_bounds():
    """
    Returns (low, high) for the steer action.
    Supports:
      - scalar range A -> (-A, +A)
      - (low, high) sequence -> (low, high)
      - [A] single-element sequence -> (-A, +A)
    Fallback to (-1, 1).
    """
    sr = STEER_RANGE
    try:
        if isinstance(sr, (int, float, np.floating)):
            a = float(abs(sr))
            return -a, a
        if isinstance(sr, (list, tuple, np.ndarray)):
            if len(sr) == 2:
                lo, hi = float(sr[0]), float(sr[1])
                if lo > hi:
                    lo, hi = hi, lo
                return lo, hi
            if len(sr) == 1:
                a = float(abs(sr[0]))
                return -a, a
    except Exception:
        pass
    return -1.0, 1.0


# --------------------
# Custom Gym Env wrapping the TinyPhysics simulator and CSV data
# --------------------
class DataSteerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, csv_path: str, sim_model: TinyPhysicsModel):
        super().__init__()
        self.csv_path = csv_path
        self.sim_model = sim_model

        # Load and pre-process data
        df = pd.read_csv(csv_path)
        self.data = pd.DataFrame({
            "roll_lataccel": np.sin(df["roll"].values) * ACC_G,
            "v_ego": df["vEgo"].values,
            "a_ego": df["aEgo"].values,
            "target_lataccel": df["targetLateralAcceleration"].values,
            # steer logged with left-positive; simulator uses right-positive
            "steer_command": -df["steerCommand"].values
        })

        # Histories/state init placeholders
        self.state_history = []
        self.action_history = []
        self.current_lataccel_history = []
        self.target_lataccel_history = []
        self.futureplan = None
        self.current_lataccel = 0.0
        self.step_idx = 0
        self.integral_error = 0.0

        # Observation is a flattened feature vector:
        # 10 channels x CONTEXT_LENGTH (past half + future half), same as original code construction
        self.n_features = 10 * CONTEXT_LENGTH
        high = np.full((self.n_features,), np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # Single continuous steer action
        self.act_low, self.act_high = _resolve_steer_bounds()
        self.action_space = gym.spaces.Box(
            low=np.array([self.act_low], dtype=np.float32),
            high=np.array([self.act_high], dtype=np.float32),
            dtype=np.float32,
        )

    def _get_state_target_future(self, idx: int):
        row = self.data.iloc[idx]
        state = {
            "roll_lataccel": row["roll_lataccel"],
            "v_ego": row["v_ego"],
            "a_ego": row["a_ego"],
        }
        target = row["target_lataccel"]
        fp_slice_end = idx + FUTURE_PLAN_STEPS
        future = {
            "lataccel": self.data["target_lataccel"].values[idx + 1:fp_slice_end].tolist(),
            "roll_lataccel": self.data["roll_lataccel"].values[idx + 1:fp_slice_end].tolist(),
            "v_ego": self.data["v_ego"].values[idx + 1:fp_slice_end].tolist(),
            "a_ego": self.data["a_ego"].values[idx + 1:fp_slice_end].tolist(),
        }
        return state, target, future

    def _pad_future(self, future: dict) -> dict:
        # Ensure future plans have at least CONTEXT_LENGTH elements
        def pad_to_len(arr: List[float], L: int):
            if len(arr) >= L:
                return arr[:L]
            last_val = arr[-1] if len(arr) > 0 else 0.0
            return arr + [last_val] * (L - len(arr))

        return {
            "lataccel": pad_to_len(future["lataccel"], CONTEXT_LENGTH),
            "roll_lataccel": pad_to_len(future["roll_lataccel"], CONTEXT_LENGTH),
            "v_ego": pad_to_len(future["v_ego"], CONTEXT_LENGTH),
            "a_ego": pad_to_len(future["a_ego"], CONTEXT_LENGTH),
        }

    def _build_obs(self) -> np.ndarray:
        # Build the 10 x CONTEXT_LENGTH matrix then flatten
        half = CONTEXT_LENGTH // 2

        # last half past signals
        roll_lataccel_hist = [s["roll_lataccel"] for s in self.state_history][-half:]
        v_ego_hist = [s["v_ego"] for s in self.state_history][-half:]
        a_ego_hist = [s["a_ego"] for s in self.state_history][-half:]
        actions_hist = self.action_history[-half:]
        cur_lat_hist = self.current_lataccel_history[-half:]
        targ_lat_hist = self.target_lataccel_history[-half:]

        # first half of future plan (already padded to CONTEXT_LENGTH)
        lataccel_fut = self.futureplan["lataccel"][:half]
        a_ego_fut = self.futureplan["a_ego"][:half]
        roll_lataccel_fut = self.futureplan["roll_lataccel"][:half]
        v_ego_fut = self.futureplan["v_ego"][:half]

        mat = np.column_stack((
            actions_hist,
            roll_lataccel_hist,
            v_ego_hist,
            a_ego_hist,
            cur_lat_hist,
            targ_lat_hist,
            lataccel_fut,
            a_ego_fut,
            roll_lataccel_fut,
            v_ego_fut,
        )).astype(np.float32)

        return mat.flatten()

    def _simulation_step(self, step_idx: int):
        # TinyPhysics prediction with clipping
        pred = self.sim_model.get_current_lataccel(
            sim_states=self.state_history[-CONTEXT_LENGTH:],
            actions=self.action_history[-CONTEXT_LENGTH:],
            past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = float(np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA))
        if step_idx >= CONTROL_START_IDX:
            self.current_lataccel = pred
        else:
            _, target_lat, _ = self._get_state_target_future(step_idx)
            self.current_lataccel = float(target_lat)

        self.current_lataccel_history.append(self.current_lataccel)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.step_idx = CONTEXT_LENGTH
        self.integral_error = 0.0

        # prime histories with first CONTEXT_LENGTH steps
        init_pack = [self._get_state_target_future(i) for i in range(self.step_idx)]
        self.state_history = [p[0] for p in init_pack]
        self.target_lataccel_history = [p[1] for p in init_pack]
        self.current_lataccel_history = [p[1] for p in init_pack]  # start with target until control start
        self.action_history = self.data["steer_command"].values[:self.step_idx].astype(np.float32).tolist()

        # set current target future plan for next step
        _, _, future = self._get_state_target_future(self.step_idx)
        self.futureplan = self._pad_future(future)
        self.current_lataccel = self.current_lataccel_history[-1]

        obs = self._build_obs()
        info = {}
        return obs.astype(np.float32), info

    def step(self, action: np.ndarray):
        # clip action to actuator range
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, self.act_low, self.act_high)
        self.action_history.append(float(action[0]))

        # advance simulation one step
        state, target, future = self._get_state_target_future(self.step_idx)
        self.state_history.append(state)
        self.target_lataccel_history.append(float(target))
        self.futureplan = self._pad_future(future)

        self._simulation_step(self.step_idx)
        self.step_idx += 1

        # reward: penalize tracking error and jerk + integral windup penalty
        current_lat = self.current_lataccel
        if len(self.current_lataccel_history) >= 2:
            jerk = (self.current_lataccel_history[-1] - self.current_lataccel_history[-2]) / DEL_T
        else:
            jerk = 0.0

        alpha = 0.01
        self.integral_error = (1 - alpha) * self.integral_error + alpha * (current_lat - float(target))
        reward = -(((current_lat - float(target)) ** 2) * 500.0 + (jerk ** 2) * 1.0) - (self.integral_error ** 2)

        terminated = self.step_idx >= len(self.data)
        truncated = False
        info = {}

        obs = self._build_obs().astype(np.float32)
        return obs, float(reward), terminated, truncated, info


# --------------------
# Training loop (CleanRL style)
# --------------------
def make_envs(args: Args, sim_model: TinyPhysicsModel):
    data_dir = args.data_dir
    csv_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")])
    if len(csv_files) < args.num_envs:
        raise RuntimeError(f"Not enough CSV files in {data_dir} for num_envs={args.num_envs}")

    def thunk(i):
        def _env():
            env = DataSteerEnv(csv_files[i], sim_model)
            # RecordEpisodeStatistics adds "final_info" with episodic returns
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env
        return _env

    return gym.vector.SyncVectorEnv([thunk(i) for i in range(args.num_envs)])


def gae_fn(next_obs, next_done, container, get_value, args: Args):
    # bootstrap value for last state
    next_value = get_value(next_obs).reshape(-1)
    lastgaelam = torch.zeros_like(next_done, dtype=torch.float32)

    dones_prev = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)

    advantages = []
    nextnonterminal = (~next_done).float()
    nextvalues = next_value
    for t in range(args.num_steps - 1, -1, -1):
        cur_val = vals_unbind[t]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - cur_val
        lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        advantages.append(lastgaelam)

        nextnonterminal = dones_prev[t]
        nextvalues = cur_val

    advantages = torch.stack(list(reversed(advantages)))
    container["advantages"] = advantages
    container["returns"] = advantages + vals
    return container


def rollout(policy, step_func, obs, done, device, args: Args, avg_returns: List[float]):
    ts = []
    for _ in range(args.num_steps):
        action, logprob, _, value = policy(obs)
        next_obs, reward, next_done, infos = step_func(action)

        if "final_info" in infos:
            for finfo in infos["final_info"]:
                if finfo is not None and "episode" in finfo:
                    r = float(finfo["episode"]["r"].reshape(()))
                    avg_returns.append(r)

        ts.append({
            "obs": obs,
            "dones": done,
            "vals": value.flatten(),
            "actions": action,
            "logprobs": logprob,
            "rewards": reward,
        })

        obs = next_obs = next_obs.to(device, non_blocking=True)
        done = next_done.to(device, non_blocking=True)

    # Stack to tensors [T, N, ...]
    container = {
        k: torch.stack([t[k] for t in ts], dim=0).to(device) for k in ts[0].keys()
    }
    return obs, done, container


def update_epoch(agent, optimizer, container_flat, args: Args, device):
    inds = torch.randperm(container_flat["obs"].shape[0], device=device).split(args.minibatch_size)
    approx_kls = []
    last_out = None

    for b in inds:
        obs_b = container_flat["obs"][b]
        actions_b = container_flat["actions"][b]
        logprobs_b = container_flat["logprobs"][b]
        adv_b = container_flat["advantages"][b]
        ret_b = container_flat["returns"][b]
        vals_b = container_flat["vals"][b]

        optimizer.zero_grad()
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs_b, actions_b)
        logratio = newlogprob - logprobs_b
        ratio = logratio.exp()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            approx_kls.append(approx_kl)

        if args.norm_adv:
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        # policy loss
        pg_loss1 = -adv_b * ratio
        pg_loss2 = -adv_b * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # value loss
        newvalue = newvalue.view(-1)
        if args.clip_vloss:
            v_loss_unclipped = (newvalue - ret_b) ** 2
            v_clipped = vals_b + torch.clamp(newvalue - vals_b, -args.clip_coef, args.clip_coef)
            v_loss_clipped = (v_clipped - ret_b) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalue - ret_b) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        loss.backward()
        gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        last_out = {
            "pg_loss": pg_loss.detach(),
            "v_loss": v_loss.detach(),
            "entropy": entropy_loss.detach(),
            "old_approx_kl": old_approx_kl.detach(),
            "approx_kl": approx_kl.detach(),
            "gn": gn.detach(),
        }

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    return last_out, torch.stack(approx_kls).mean() if approx_kls else torch.tensor(0.0, device=device)


def main(args: Args):
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Compute batch sizes
    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size

    # Load TinyPhysics once and share across envs
    sim_model = TinyPhysicsModel(args.model_path, debug=False)

    # Vectorized envs
    envs = make_envs(args, sim_model)
    n_obs = int(np.prod(envs.single_observation_space.shape))
    n_act = int(np.prod(envs.single_action_space.shape))
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    # Agent and optimizer
    agent = Agent(n_obs, n_act, device=device).to(device)
    agent_inference = Agent(n_obs, n_act, device=device).to(device)
    agent_inference.load_state_dict(agent.state_dict())

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Executables
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    # Step function that interacts with vector envs
    def step_func(action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        next_obs_np, reward, terminated, truncated, infos = envs.step(action.detach().cpu().numpy())
        next_done = np.logical_or(terminated, truncated)
        return (
            torch.as_tensor(next_obs_np, dtype=torch.float32, device=device),
            torch.as_tensor(reward, dtype=torch.float32, device=device),
            torch.as_tensor(next_done, dtype=torch.bool, device=device),
            infos,
        )

    # Reset
    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)

    avg_returns: List[float] = []
    global_step = 0
    pbar = range(1, args.num_iterations + 1)

    for iteration in pbar:
        # Anneal LR
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            for pg in optimizer.param_groups:
                pg["lr"] = lrnow

        # Sync inference weights from learner
        agent_inference.load_state_dict(agent.state_dict())

        # Rollout
        t0 = time.time()
        next_obs, next_done, container = rollout(policy, step_func, next_obs, next_done, device, args, avg_returns)
        global_step += args.num_envs * args.num_steps

        # Convert container to tensors with expected keys
        container_td = {
            "obs": container["obs"],                # [T, N, obs]
            "dones": container["dones"],            # [T, N]
            "vals": container["vals"],              # [T, N]
            "actions": container["actions"],        # [T, N, act]
            "logprobs": container["logprobs"],      # [T, N]
            "rewards": container["rewards"],        # [T, N]
        }

        # GAE
        # Flatten obs for value function
        container_td["obs_flat"] = container_td["obs"].view(-1, n_obs)
        next_obs_for_gae = next_obs.view(args.num_envs, n_obs)
        container_td_torch = {k: v for k, v in container_td.items()}
        container_td_torch = gae_fn(
            next_obs_for_gae, next_done, container_td_torch, get_value=lambda x: get_value(x), args=args
        )

        # Flatten the batch [T, N] -> [T*N]
        def flatten_key(k):
            return container_td_torch[k].reshape(-1, *container_td_torch[k].shape[2:]) if container_td_torch[k].dim() > 2 else container_td_torch[k].reshape(-1)

        container_flat = {
            "obs": container_td_torch["obs"].reshape(-1, n_obs),
            "actions": container_td_torch["actions"].reshape(-1, n_act),
            "logprobs": flatten_key("logprobs"),
            "advantages": flatten_key("advantages"),
            "returns": flatten_key("returns"),
            "vals": flatten_key("vals"),
        }

        # Optimize
        for _ in range(args.update_epochs):
            out, approx_kl = update_epoch(agent, optimizer, container_flat, args, device)
            if args.target_kl is not None and approx_kl.item() > args.target_kl:
                break

        # Logging (stdout)
        if (iteration % 10) == 0:
            sps = int((args.num_envs * args.num_steps) / max(1e-6, (time.time() - t0)))
            avg_ret = np.mean(avg_returns[-20:]) if len(avg_returns) > 0 else 0.0
            print(f"[it {iteration:04d}] step={global_step} sps={sps} "
                  f"avg_ep_ret={avg_ret:6.2f} pg={out['pg_loss']:.4f} "
                  f"v={out['v_loss']:.4f} ent={out['entropy']:.4f} kl={out['approx_kl']:.6f}")

    envs.close()
    # Save final weights
    torch.save(agent.state_dict(), "ppo_cleanrl_style_final.pth")


if __name__ == "__main__":
    args = Args()
    main(args)