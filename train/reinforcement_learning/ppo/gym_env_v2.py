import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import pandas as pd
from collections import namedtuple
from typing import Tuple

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

State = namedtuple("State", ["roll_lataccel", "v_ego", "a_ego"])
FuturePlan = namedtuple("FuturePlan", ["lataccel", "roll_lataccel", "v_ego", "a_ego"])


class PPOEnv(gym.Env):
    """
    CleanRL-compatible TinyPhysics PPO environment.
    Returns continuous observation tensors and scalar rewards.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        simulation_model: TinyPhysicsModel,
        data_path: str,
        debug: bool = False,
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sim_model = simulation_model
        self.data_path = data_path
        self.debug = debug
        self.data = self._get_data(data_path)

        self.integral_error = 0.0

        # Observation space (flattened CONTEXT_LENGTH/2 x 10 input)
        obs_dim = int((CONTEXT_LENGTH / 2) * 10)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: continuous steering
        self.action_space = spaces.Box(
            low=np.array([STEER_RANGE[0]], dtype=np.float32),
            high=np.array([STEER_RANGE[1]], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset()

    # -------------------------------
    # Internal Data Handling
    # -------------------------------

    def _get_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        return pd.DataFrame(
            {
                "roll_lataccel": np.sin(df["roll"].values) * ACC_G,
                "v_ego": df["vEgo"].values,
                "a_ego": df["aEgo"].values,
                "target_lataccel": df["targetLateralAcceleration"].values,
                "steer_command": -df["steerCommand"].values,
            }
        )

    def _get_state_target_futureplan(self, step_idx: int) -> Tuple[State, float, FuturePlan]:
        state = self.data.iloc[step_idx]
        return (
            State(
                roll_lataccel=state["roll_lataccel"],
                v_ego=state["v_ego"],
                a_ego=state["a_ego"],
            ),
            state["target_lataccel"],
            FuturePlan(
                lataccel=self.data["target_lataccel"].values[
                    step_idx + 1 : step_idx + FUTURE_PLAN_STEPS
                ].tolist(),
                roll_lataccel=self.data["roll_lataccel"].values[
                    step_idx + 1 : step_idx + FUTURE_PLAN_STEPS
                ].tolist(),
                v_ego=self.data["v_ego"].values[
                    step_idx + 1 : step_idx + FUTURE_PLAN_STEPS
                ].tolist(),
                a_ego=self.data["a_ego"].values[
                    step_idx + 1 : step_idx + FUTURE_PLAN_STEPS
                ].tolist(),
            ),
        )

    def _simulation_step(self):
        pred = self.sim_model.get_current_lataccel(
            sim_states=self.state_history[-CONTEXT_LENGTH:],
            actions=self.action_history[-CONTEXT_LENGTH:],
            past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:],
        )
        pred = np.clip(
            pred,
            self.current_lataccel - MAX_ACC_DELTA,
            self.current_lataccel + MAX_ACC_DELTA,
        )
        if self.step_idx >= CONTROL_START_IDX:
            self.current_lataccel = pred
        else:
            self.current_lataccel = self._get_state_target_futureplan(self.step_idx)[1]

        self.current_lataccel_history.append(self.current_lataccel)

    def _build_observation(self):
        last_action = self.action_history[-1]
        roll_lataccel = [s.roll_lataccel for s in self.state_history[-CONTEXT_LENGTH:]]
        v_ego = [s.v_ego for s in self.state_history[-CONTEXT_LENGTH:]]
        a_ego = [s.a_ego for s in self.state_history[-CONTEXT_LENGTH:]]
        target_lataccel = self.target_lataccel_history[-CONTEXT_LENGTH:]
        current_lataccel = self.current_lataccel_history[-CONTEXT_LENGTH:]

        # Pad future plan to match CONTEXT_LENGTH/2
        f = self.futureplan
        pad = int(CONTEXT_LENGTH / 2)
        for name in ["lataccel", "roll_lataccel", "v_ego", "a_ego"]:
            seq = getattr(f, name)
            if len(seq) < pad:
                seq.extend([seq[-1] if seq else 0.0] * (pad - len(seq)))

        input_np = np.column_stack(
            (
                self.action_history[-pad:],
                roll_lataccel[-pad:],
                v_ego[-pad:],
                a_ego[-pad:],
                current_lataccel[-pad:],
                target_lataccel[-pad:],
                f.lataccel[:pad],
                f.a_ego[:pad],
                f.roll_lataccel[:pad],
                f.v_ego[:pad],
            )
        ).flatten()

        return input_np.astype(np.float32)

    # -------------------------------
    # Gym API
    # -------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = CONTEXT_LENGTH
        self.integral_error = 0.0
        self.state_history = []
        self.action_history = self.data["steer_command"].values[:CONTEXT_LENGTH].tolist()
        self.current_lataccel_history = []
        self.target_lataccel_history = []

        for i in range(CONTEXT_LENGTH):
            s, t, f = self._get_state_target_futureplan(i)
            self.state_history.append(s)
            self.current_lataccel_history.append(t)
            self.target_lataccel_history.append(t)
        self.futureplan = f
        self.current_lataccel = self.current_lataccel_history[-1]

        obs = self._build_observation()
        return obs, {}

    def step(self, action):
        action = float(np.clip(action, STEER_RANGE[0], STEER_RANGE[1]))
        self.action_history.append(action)
        s, target, f = self._get_state_target_futureplan(self.step_idx)
        self.state_history.append(s)
        self.target_lataccel_history.append(target)
        self.futureplan = f

        # Simulate
        self._simulation_step()

        current_lataccel = self.current_lataccel
        jerk = (
            (self.current_lataccel_history[-1] - self.current_lataccel_history[-2]) / DEL_T
            if len(self.current_lataccel_history) >= 2
            else 0.0
        )

        # Reward shaping
        alpha = 0.01
        self.integral_error = (1 - alpha) * self.integral_error + alpha * (
            current_lataccel - target
        )
        reward = -(
            (current_lataccel - target) ** 2 * 500 + jerk**2
        ) - self.integral_error**2

        self.step_idx += 1
        terminated = self.step_idx >= len(self.data)
        truncated = False

        obs = self._build_observation()
        info = {
            "cost": ((current_lataccel - target) ** 2 * 5000 + jerk**2 * 100),
            "integral_error": self.integral_error,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
