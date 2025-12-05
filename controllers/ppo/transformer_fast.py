from controllers import BaseController
import numpy as np
import onnxruntime as ort
from tinyphysics import CONTEXT_LENGTH, CONTROL_START_IDX


class Controller(BaseController):
    """
    Ultra-optimized PPO transformer controller:
      • ONNX Runtime for fast inference
      • no SciPy
      • no PyTorch during inference
      • cheap IIR filters instead of filtfilt
      • preallocated 100-dim observation vector
      • zero allocations in update() 
    """

    def __init__(self):
        # PID parameters
        self.p = 0.1
        self.i = 0.9
        self.d = -0.003
        self.alpha = 0.6
        self.error_integral = 0.0
        self.prev_error = 0.0

        # --- FAST LOW-PASS FILTER STATES ---
        self.lp_target = 0.0
        self.lp_action = 0.0
        self.alpha_target = 0.15
        self.alpha_action = 0.20

        # --- LOAD ONNX transformer model ---
        self.ort = ort.InferenceSession(
            "./models/ppo_transformer_controller.onnx",
            providers=["CPUExecutionProvider"],
        )

        self.value = []
        self.mean = []
        self.pid_action = []

        # --- Preallocated 100-dim observation buffer ---
        self.obs_vec = np.zeros(100, dtype=np.float32)

        # Number of timesteps = 50 past + 50 future
        self.T = CONTEXT_LENGTH // 2

    # ultra-fast IIR filters
    def _lp_target(self, x):
        self.lp_target = self.alpha_target * x + (1 - self.alpha_target) * self.lp_target
        return self.lp_target

    def _lp_action(self, x):
        self.lp_action = self.alpha_action * x + (1 - self.alpha_action) * self.lp_action
        return self.lp_action

    # main control update
    def update(self, target_lataccel_history, current_lataccel_history, state_history, action_history, future_plan):

        T = self.T

        # extract state features (last T samples)
        a_ego = np.array([s.a_ego for s in state_history[-T:]], dtype=np.float32)
        v_ego = np.array([s.v_ego for s in state_history[-T:]], dtype=np.float32)
        roll_lataccel = np.array([s.roll_lataccel for s in state_history[-T:]], dtype=np.float32)

        curr_lat = np.array(current_lataccel_history[-T:], dtype=np.float32)
        tgt_lat = np.array(target_lataccel_history[-T:], dtype=np.float32)
        act_hist = np.array(action_history[-T:], dtype=np.float32)

        # pad future plan if needed
        if len(future_plan.lataccel) < T:
            pad = T - len(future_plan.lataccel)

            def pad_list(lst):
                last = lst[-1] if len(lst) else 0.0
                lst.extend([last] * pad)

            pad_list(future_plan.lataccel)
            pad_list(future_plan.roll_lataccel)
            pad_list(future_plan.v_ego)
            pad_list(future_plan.a_ego)

        f_lat = np.array(future_plan.lataccel[:T], dtype=np.float32)
        f_ae  = np.array(future_plan.a_ego[:T], dtype=np.float32)
        f_rl  = np.array(future_plan.roll_lataccel[:T], dtype=np.float32)
        f_ve  = np.array(future_plan.v_ego[:T], dtype=np.float32)

        # ------------------------------------------------------------
        # Preallocated 100-dim observation construction (FAST)
        # ------------------------------------------------------------
        # Order must match training!
        self.obs_vec[:T]      = act_hist
        self.obs_vec[T:2*T]   = roll_lataccel
        self.obs_vec[2*T:3*T] = v_ego
        self.obs_vec[3*T:4*T] = a_ego
        self.obs_vec[4*T:5*T] = curr_lat
        self.obs_vec[5*T:6*T] = tgt_lat
        self.obs_vec[6*T:7*T] = f_lat
        self.obs_vec[7*T:8*T] = f_ae
        self.obs_vec[8*T:9*T] = f_rl
        self.obs_vec[9*T:10*T] = f_ve

        # prepare ONNX input
        inp = self.obs_vec.reshape(1, 100)

        # ------------------------------------------------------------
        # ONNX INFERENCE (super fast)
        # ------------------------------------------------------------
        mean, std, value = self.ort.run(None, {"obs": inp})
        mean = float(mean[0][0])
        value = float(value[0][0])

        self.value.append(value)
        self.mean.append(mean)

        # ------------------------------------------------------------
        # FAST TARGET LATACCEL FILTER
        # ------------------------------------------------------------
        filtered_target = self._lp_target(tgt_lat[-1])

        # blend with short future plan horizon
        if len(future_plan.lataccel) >= 5:
            filtered_target = np.average(
                [filtered_target] + list(future_plan.lataccel[:5]),
                weights=[4, 3, 2, 2, 2, 1]
            )

        # ------------------------------------------------------------
        # PID CONTROL (fast, no additional allocations)
        # ------------------------------------------------------------
        error = filtered_target - curr_lat[-1]

        if len(tgt_lat) < CONTROL_START_IDX:
            self.error_integral = 0.0
        else:
            self.error_integral += error * 0.1

        error_diff = (error - self.prev_error) / 0.1
        self.prev_error = error

        pid = self.p * error + self.i * self.error_integral + self.d * error_diff
        self.pid_action.append(pid)

        # ------------------------------------------------------------
        # Final action: PPO mean + PID
        # ------------------------------------------------------------
        action = self.alpha * mean + pid

        # ------------------------------------------------------------
        # FAST ACTION FILTER (no SciPy)
        # ------------------------------------------------------------
        action = self._lp_action(action)

        # extend history
        action_history.append(action)

        return action
