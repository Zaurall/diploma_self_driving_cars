import pandas as pd
import numpy as np
import torch
from tinyphysics import TinyPhysicsModel, CONTROL_START_IDX, STEER_RANGE, CONTEXT_LENGTH, DEL_T, FUTURE_PLAN_STEPS, MAX_ACC_DELTA, ACC_G
from collections import namedtuple
from typing import  Tuple

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])

class PPOEnv:
    def __init__(self, simulation_model: TinyPhysicsModel, data_path: str, policy, debug: bool = False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = data_path
        # TODO
        self.sim_model = simulation_model
        self.data = self.get_data(data_path)
        self.policy = policy.to(self.device)  # Assuming 3 input features: roll_lataccel, v_ego, a_ego
        self.debug = debug
        self.reset()
        self.integral_error = 0
        # Single continuous steer action

    def get_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        processed_df = pd.DataFrame({
            'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
            'v_ego': df['vEgo'].values,
            'a_ego': df['aEgo'].values,
            'target_lataccel': df['targetLateralAcceleration'].values,
            'steer_command': -df['steerCommand'].values  # steer commands are logged with left-positive convention but this simulator uses right-positive
        })
        return processed_df  
        
    def reset(self) -> None:
        # TODO maybe add previous steps
        self.step_idx = CONTEXT_LENGTH
        state_target_futureplans = [self.get_state_target_futureplan(i) for i in range(self.step_idx)]
        self.state_history = [x[0] for x in state_target_futureplans]
        self.action_history = self.data['steer_command'].values[:self.step_idx].tolist()
        self.current_lataccel_history = [x[1] for x in state_target_futureplans]
        self.target_lataccel_history = [x[1] for x in state_target_futureplans]
        self.target_future = None
        self.current_lataccel = self.current_lataccel_history[-1]

    def get_state_target_futureplan(self, step_idx: int) -> Tuple[State, float, FuturePlan]:
        state = self.data.iloc[step_idx]
        return (
            State(roll_lataccel=state['roll_lataccel'], v_ego=state['v_ego'], a_ego=state['a_ego']),
            state['target_lataccel'],
            FuturePlan(
                # TODO maybe not + future steps, but 1+future_steps
                lataccel=self.data['target_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
                roll_lataccel=self.data['roll_lataccel'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
                v_ego=self.data['v_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist(),
                a_ego=self.data['a_ego'].values[step_idx + 1:step_idx + FUTURE_PLAN_STEPS].tolist()
            )
        )
        
    def simulation_step(self, step_idx: int) -> None:
        pred = self.sim_model.get_current_lataccel(
            sim_states=self.state_history[-CONTEXT_LENGTH:],
            actions=self.action_history[-CONTEXT_LENGTH:],
            past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = np.clip(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
        if step_idx >= CONTROL_START_IDX:
            self.current_lataccel = pred
        else:
            self.current_lataccel = self.get_state_target_futureplan(step_idx)[1]

        self.current_lataccel_history.append(self.current_lataccel)

    def normalize(self, val, min_val, max_val):
        # Maps value to [-1, 1]
        return 2 * (val - min_val) / (max_val - min_val) - 1

    def step(self, evaluation=False) -> None:
        state, target, futureplan = self.get_state_target_futureplan(self.step_idx)
        self.state_history.append(state)
        self.target_lataccel_history.append(target)
        self.futureplan = futureplan

        # --- 1. PREPARE DATA ---
        # Use full CONTEXT_LENGTH, not half
        ctx_len = CONTEXT_LENGTH 
        
        # Get raw lists
        actions = np.array(self.action_history[-ctx_len:])
        rolls = np.array([s.roll_lataccel for s in self.state_history[-ctx_len:]])
        v_egos = np.array([s.v_ego for s in self.state_history[-ctx_len:]])
        a_egos = np.array([s.a_ego for s in self.state_history[-ctx_len:]])
        current_lats = np.array(self.current_lataccel_history[-ctx_len:])
        target_lats = np.array(self.target_lataccel_history[-ctx_len:])
        
        # Handle Future Plan Padding (Same as your code)
        if len(self.futureplan.lataccel) < ctx_len:
            pad = ctx_len - len(self.futureplan.lataccel)
            self.futureplan.lataccel.extend([self.futureplan.lataccel[-1]] * pad)
            self.futureplan.roll_lataccel.extend([self.futureplan.roll_lataccel[-1]] * pad)
            self.futureplan.v_ego.extend([self.futureplan.v_ego[-1]] * pad)
            self.futureplan.a_ego.extend([self.futureplan.a_ego[-1]] * pad)

        # --- 2. NORMALIZE INPUTS ---
        # Approximate ranges based on dataset
        norm_actions = self.normalize(actions, -2.0, 2.0)
        norm_rolls = self.normalize(rolls, -0.2, 0.2) # Roll is usually small
        norm_v_ego = self.normalize(v_egos, 0.0, 35.0) # 0 to 35 m/s
        norm_a_ego = self.normalize(a_egos, -4.0, 4.0)
        norm_curr_lat = self.normalize(current_lats, -5.0, 5.0)
        norm_targ_lat = self.normalize(target_lats, -5.0, 5.0)
        
        norm_fut_lat = self.normalize(np.array(self.futureplan.lataccel[:ctx_len]), -5.0, 5.0)
        norm_fut_roll = self.normalize(np.array(self.futureplan.roll_lataccel[:ctx_len]), -0.2, 0.2)
        
        # Stack normalized features
        input_np = np.column_stack((
            norm_actions,
            norm_rolls,
            norm_v_ego,
            norm_a_ego,
            norm_curr_lat,
            norm_targ_lat,
            norm_fut_lat,
            norm_fut_roll
        ))
        
        input_tensor = torch.tensor(input_np, dtype=torch.float32).flatten().unsqueeze(0).to(self.device)
        
        # print(input_tensor.shape)
        
        # Get action distribution from policy
        mean, std, value = self.policy(input_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if evaluation:
            action = dist.mean.item()
        else:
            action = action.item()
        self.action_history.append(action)
        self.simulation_step(self.step_idx)
        self.step_idx += 1

         # Compute reward (goal: match target lataccel)
        current_lataccel = self.current_lataccel
        if len(self.current_lataccel_history) >= 2:
            jerk = (self.current_lataccel_history[-1] - self.current_lataccel_history[-2]) / DEL_T
        else:
            jerk = 0.0
        # error = abs(current_lataccel - target)
        # if error < 0.05:
        #     reward = 0
        # elif error < 2:
        #     # Linearly decreasing reward between 0.5 and max_error
        #     reward = - (error - 0.05) / (2 - 0.5)
        # else:
        #     reward = -10.0
        alpha=0.01
        self.integral_error =(1-alpha)*self.integral_error+alpha*(current_lataccel - target) 
        reward = -((current_lataccel - target)**2 * 500 + jerk**2 * 1)-self.integral_error**2 
        # Done condition
        done = self.step_idx >= len(self.data)

        # Optional: store experience for training
        self.rollout_buffer.append({
            "obs": input_tensor,
            "action": action,
            "log_prob": log_prob.item(),
            "value": value.item(),
            "reward": reward,
            "done": done
        })
        cost=((current_lataccel - target)**2 * 5000 + jerk**2 * 100)
        return cost, value.item(), log_prob.item(), done
