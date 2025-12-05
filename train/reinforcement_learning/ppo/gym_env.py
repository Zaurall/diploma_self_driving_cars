import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import pandas as pd
from collections import namedtuple
from tinyphysics import (
    TinyPhysicsModel,
    CONTROL_START_IDX,
    STEER_RANGE,
    CONTEXT_LENGTH,
    DEL_T,
    FUTURE_PLAN_STEPS,
    MAX_ACC_DELTA,
    ACC_G
)

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])


class PPOEnv(gym.Env):
    """
    A LearnRL-compatible gymnasium environment for PPO with continuous action space.
    """

    metadata = {"render_modes": []}

    def __init__(self, simulation_model: TinyPhysicsModel, data_path: str, debug: bool = False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sim_model = simulation_model
        self.data = self._load_data(data_path)
        self.debug = debug

        # Environment-specific parameters
        self.integral_error = 0.0
        self.step_idx = 0

        # Define continuous action space (steer angle)
        self.action_space = spaces.Box(
            low=np.array([-STEER_RANGE], dtype=np.float32),
            high=np.array([STEER_RANGE], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        # Define observation space (flattened state vector)
        obs_dim = int(CONTEXT_LENGTH * 10 / 2)  # 10 variables, half context window
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Internal buffers
        self.state_history = []
        self.action_history = []
        self.current_lataccel_history = []
        self.target_lataccel_history = []
        self.futureplan = None
        self.current_lataccel = 0.0

        self.reset(seed=None)

    def _load_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        return pd.DataFrame({
            'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
            'v_ego': df['vEgo'].values,
            'a_ego': df['aEgo'].values,
            'target_lataccel': df['targetLateralAcceleration'].values,
            'steer_command': -df['steerCommand'].values  # sign convention fix
        })

    def _get_state_target_futureplan(self, step_idx: int):
        state = self.data.iloc[step_idx]
        return (
            State(roll_lataccel=state['roll_lataccel'], v_ego=state['v_ego'], a_ego=state['a_ego']),
            state['target_lataccel'],
            FuturePlan(
                lataccel=self.data['target_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
                roll_lataccel=self.data['roll_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
                v_ego=self.data['v_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
                a_ego=self.data['a_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist()
            )
        )

    def _simulation_step(self, step_idx: int):
        pred = self.sim_model.get_current_lataccel(
            sim_states=self.state_history[-CONTEXT_LENGTH:],
            actions=self.action_history[-CONTEXT_LENGTH:],
            past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
        if step_idx >= CONTROL_START_IDX:
            self.current_lataccel = pred
        else:
            self.current_lataccel = self._get_state_target_futureplan(step_idx)[1]
        self.current_lataccel_history.append(self.current_lataccel)

    def _get_obs(self):
        """Builds and flattens the observation vector."""
        roll_lataccel = [s.roll_lataccel for s in self.state_history[-CONTEXT_LENGTH:]]
        v_ego = [s.v_ego for s in self.state_history[-CONTEXT_LENGTH:]]
        a_ego = [s.a_ego for s in self.state_history[-CONTEXT_LENGTH:]]

        # Handle future plan padding
        if len(self.futureplan.lataccel) < CONTEXT_LENGTH:
            pad_len = CONTEXT_LENGTH - len(self.futureplan.lataccel)
            for attr in ["lataccel", "roll_lataccel", "v_ego", "a_ego"]:
                seq = getattr(self.futureplan, attr)
                seq.extend([seq[-1] if seq else 0.0] * pad_len)

        input_np = np.column_stack((
            self.action_history[-int(CONTEXT_LENGTH/2):],
            roll_lataccel[-int(CONTEXT_LENGTH/2):],
            v_ego[-int(CONTEXT_LENGTH/2):],
            a_ego[-int(CONTEXT_LENGTH/2):],
            self.current_lataccel_history[-int(CONTEXT_LENGTH/2):],
            self.target_lataccel_history[-int(CONTEXT_LENGTH/2):],
            self.futureplan.lataccel[:int(CONTEXT_LENGTH/2)],
            self.futureplan.a_ego[:int(CONTEXT_LENGTH/2)],
            self.futureplan.roll_lataccel[:int(CONTEXT_LENGTH/2)],
            self.futureplan.v_ego[:int(CONTEXT_LENGTH/2)],
        ))
        return input_np.flatten().astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = CONTEXT_LENGTH
        state_target_futureplans = [self._get_state_target_futureplan(i) for i in range(self.step_idx)]
        self.state_history = [x[0] for x in state_target_futureplans]
        self.action_history = self.data['steer_command'].values[:self.step_idx].tolist()
        self.current_lataccel_history = [x[1] for x in state_target_futureplans]
        self.target_lataccel_history = [x[1] for x in state_target_futureplans]
        self.current_lataccel = self.current_lataccel_history[-1]
        self.futureplan = self._get_state_target_futureplan(self.step_idx)[2]
        self.integral_error = 0.0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = float(np.clip(action, -STEER_RANGE, STEER_RANGE))
        self.action_history.append(action)
        self._simulation_step(self.step_idx)

        state, target, futureplan = self._get_state_target_futureplan(self.step_idx)
        self.state_history.append(state)
        self.target_lataccel_history.append(target)
        self.futureplan = futureplan

        current_lataccel = self.current_lataccel
        jerk = (
            (self.current_lataccel_history[-1] - self.current_lataccel_history[-2]) / DEL_T
            if len(self.current_lataccel_history) >= 2 else 0.0
        )

        # Reward: penalize deviation and jerk
        alpha = 0.01
        self.integral_error = (1 - alpha) * self.integral_error + alpha * (current_lataccel - target)
        reward = -((current_lataccel - target)**2 * 500 + jerk**2) - self.integral_error**2

        self.step_idx += 1
        terminated = self.step_idx >= len(self.data) - FUTURE_PLAN_STEPS
        truncated = False

        obs = self._get_obs()
        info = {"cost": ((current_lataccel - target)**2 * 5000 + jerk**2 * 100)}

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
