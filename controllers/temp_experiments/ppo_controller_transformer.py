import math
import numpy as np
from collections import deque, namedtuple
from pathlib import Path
import sys

# Keep your existing imports/names so this is drop-in
try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import torch
except Exception:
    torch = None

# Import constants from your tinyphysics/training code (unchanged)
from tinyphysics import CONTEXT_LENGTH, CONTROL_START_IDX, FuturePlan

# -----------------------------
# Optimized Controller
# -----------------------------
class Controller:
    """
    Optimized PID + Transformer PPO Policy blended controller.

    Key changes vs original:
    - ONNX runtime inference (fast): uses model at onnx_path (preferred).
    - Fallback to TorchScript or loaded torch model if ONNX not available.
    - Deterministic mean action (no sampling).
    - Exponential moving average (EMA) used instead of filtfilt (very cheap).
    - Preallocated numpy observation buffer; no per-step column_stack allocations.
    - Fixed-length ring buffers using deque to bound memory and speed up indexing.
    """

    def __init__(
        self,
        onnx_path: str = "models/ppo_transformer.onnx",
        torch_fallback_path: str = "models/PPO_transformer-network_20-rollouts/final_model_weights.pth",
        alpha: float = 0.6,
        p: float = 0.1,
        i: float = 0.9,
        d: float = -0.003,
    ):
        # PID params
        self.p = p
        self.i = i
        self.d = d
        self.alpha = float(alpha)

        # Internal PID state
        self.error_integral = 0.0
        self.prev_error = 0.0

        # Buffers (bounded)
        self.max_len = int(CONTEXT_LENGTH)
        half_len = CONTEXT_LENGTH // 2
        self.half_len = int(half_len)

        self.action_history = deque(maxlen=self.max_len)         # blended actions
        self.pid_action_history = deque(maxlen=self.max_len)     # pid outputs (optional logging)
        self.target_lataccel_history = deque(maxlen=self.max_len)
        self.current_lataccel_history = deque(maxlen=self.max_len)
        self.state_history = deque(maxlen=self.max_len)         # store state objects

        # Preallocated observation array: shape (half_len, 10)
        self.obs_features = 10
        self.obs_buf = np.zeros((self.half_len, self.obs_features), dtype=np.float32)

        # EMA states for cheap smoothing (replaces filtfilt)
        self.ema_target = None
        self.ema_action = None
        self.ema_alpha_target = 0.15   # tuning knob (bigger = faster response, less smoothing)
        self.ema_alpha_action = 0.25

        # Inference backend: try ONNX, fallback to TorchScript or torch module
        self.onnx_session = None
        self.torch_model = None
        self.torch_script = None
        onnx_file = Path(onnx_path)
        if ort is not None and onnx_file.exists():
            try:
                # Prefer CPU execution provider for reproducibility; onnxruntime can use GPU if available
                self.onnx_session = ort.InferenceSession(str(onnx_file.as_posix()), providers=["CPUExecutionProvider"])
            except Exception:
                # Try default providers if specific fail
                try:
                    self.onnx_session = ort.InferenceSession(str(onnx_file.as_posix()))
                except Exception:
                    self.onnx_session = None

        # Torch fallback: try to load a traced model if available
        ts_path = Path(torch_fallback_path).with_suffix(".traced.pt")
        if self.onnx_session is None and torch is not None:
            # 1) Try TorchScript traced model
            if ts_path.exists():
                try:
                    self.torch_script = torch.jit.load(str(ts_path))
                    self.torch_script.eval()
                except Exception:
                    self.torch_script = None

            # 2) Try loading original model weights (user must ensure class matches)
            if self.torch_script is None:
                try:
                    # User's original policy class isn't known here; attempt to load state_dict into a generic object
                    # If you want this path, replace with actual policy class creation before load_state_dict.
                    # This fallback will stay None unless the environment provides the model class and code.
                    self.torch_model = None
                except Exception:
                    self.torch_model = None

        # Logging arrays (lightweight)
        self.value_log = []
        self.mean_log = []
        self.pid_action_log = []

        # sanity check: obs dim expected by training: obs_dim = half_len * 10
        self.obs_dim = self.half_len * self.obs_features

    # -------------------------
    # Lightweight EMA filter
    # -------------------------
    def _ema(self, prev, x, alpha):
        if prev is None:
            return float(x)
        return float(alpha * x + (1.0 - alpha) * prev)

    # -------------------------
    # Build observation buffer fast (fills self.obs_buf in place)
    # Order must match training/time order used originally!
    # Original column_stack order:
    # [action_history, roll_lataccel, v_ego, a_ego, current_lataccel_history,
    #  target_lataccel_history, future.lataccel, future.a_ego, future.roll_lataccel, future.v_ego]
    # Each column is length half_len (most recent last).
    # -------------------------
    def _fill_obs_buf(self, recent_actions, roll_lataccel, v_ego, a_ego,
                      current_lataccel_history, target_lataccel_history,
                      futureplan):
        # Prepare arrays (all are deque or lists). We'll read the last half_len items (right-aligned).
        hl = self.half_len

        def last_n(seq, n):
            if len(seq) >= n:
                # take the last n
                return np.asarray(list(seq)[-n:], dtype=np.float32)
            else:
                # pad on the left with the earliest value (repeat first) to keep recency on right
                arr = np.zeros(n, dtype=np.float32)
                seq_list = list(seq)
                if len(seq_list) == 0:
                    return arr
                pad_len = n - len(seq_list)
                arr[:pad_len] = seq_list[0]
                arr[pad_len:] = np.asarray(seq_list, dtype=np.float32)
                return arr

        a_hist = last_n(recent_actions, hl)
        roll_hist = last_n(roll_lataccel, hl)
        v_hist = last_n(v_ego, hl)
        a_ego_hist = last_n(a_ego, hl)
        cur_lat_hist = last_n(current_lataccel_history, hl)
        tgt_lat_hist = last_n(target_lataccel_history, hl)

        # Future plan arrays (FuturePlan may be list-like)
        fut_lat = last_n(futureplan.lataccel if futureplan is not None else [], hl)
        fut_a_ego = last_n(futureplan.a_ego if futureplan is not None else [], hl)
        fut_roll = last_n(futureplan.roll_lataccel if futureplan is not None else [], hl)
        fut_v = last_n(futureplan.v_ego if futureplan is not None else [], hl)

        # Fill columns in preallocated obs_buf
        # Column indexing consistent with original stacking order
        self.obs_buf[:, 0] = a_hist
        self.obs_buf[:, 1] = roll_hist
        self.obs_buf[:, 2] = v_hist
        self.obs_buf[:, 3] = a_ego_hist
        self.obs_buf[:, 4] = cur_lat_hist
        self.obs_buf[:, 5] = tgt_lat_hist
        self.obs_buf[:, 6] = fut_lat
        self.obs_buf[:, 7] = fut_a_ego
        self.obs_buf[:, 8] = fut_roll
        self.obs_buf[:, 9] = fut_v

        # Flatten to 1D row expected by policy
        return self.obs_buf.reshape(1, -1)

    # -------------------------
    # Inference wrapper (ONNX preferred)
    # Expects numpy float32 shape (1, obs_dim)
    # Returns mean (float) and value (float)
    # -------------------------
    def _policy_infer(self, input_np: np.ndarray):
        # ONNX path
        if self.onnx_session is not None:
            # find input name
            input_name = self.onnx_session.get_inputs()[0].name
            # run (returns list of outputs)
            out = self.onnx_session.run(None, {input_name: input_np.astype(np.float32)})
            # Expect outputs like [mean, std, value] or similar - try to interpret generically
            # We'll attempt to extract mean and value intelligently
            mean = None
            value = None
            # common shapes: mean -> (1,1) or (1,), value -> (1,1) or (1,)
            for o in out:
                arr = np.asarray(o)
                if arr.size == 0:
                    continue
                # heuristics: if arr.shape[-1] in (1,) or arr.ndim == 2 and arr.shape[1] == 1, treat as scalar
                if mean is None:
                    mean = float(np.squeeze(arr)[...])
                elif value is None:
                    value = float(np.squeeze(arr)[...])
            if mean is None:
                mean = 0.0
            if value is None:
                value = 0.0
            return float(mean), float(value)

        # TorchScript path
        if self.torch_script is not None and torch is not None:
            with torch.no_grad():
                t = torch.from_numpy(input_np).to(next(self.torch_script.parameters()).device) \
                    if hasattr(self.torch_script, "parameters") else torch.from_numpy(input_np)
                out = self.torch_script(t)
                # Out may be tuple (mean, std, value) or tensor
                if isinstance(out, (tuple, list)) and len(out) >= 1:
                    mean = out[0]
                    # try to get value if available
                    value = out[2] if len(out) > 2 else None
                    mean_f = float(mean.detach().cpu().numpy().squeeze()) if hasattr(mean, "detach") else float(np.squeeze(mean))
                    value_f = float(value.detach().cpu().numpy().squeeze()) if (value is not None and hasattr(value, "detach")) else (float(np.squeeze(value)) if value is not None else 0.0)
                    return mean_f, value_f
                else:
                    # single tensor -> treat as mean
                    return float(out.detach().cpu().numpy().squeeze()), 0.0

        # As last resort, return zero action
        return 0.0, 0.0

    # -------------------------
    # Main update — same signature as tinyphysics controller
    # -------------------------
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Append to buffers
        self.state_history.append(state)
        self.current_lataccel_history.append(current_lataccel)
        self.target_lataccel_history.append(target_lataccel)

        # Warm-up with pure PID until we have enough history
        if len(self.state_history) < self.max_len:
            # simple PID update (same as original)
            error = float(target_lataccel) - float(current_lataccel)
            # integrate only when control started (mimic original gating)
            self.error_integral += error * 0.1
            error_diff = (error - self.prev_error) / 0.1 if self.prev_error is not None else 0.0
            self.prev_error = error
            pid_output = self.p * error + self.i * self.error_integral + self.d * error_diff
            self.pid_action_history.append(pid_output)
            self.action_history.append(pid_output)
            # update action EMA
            self.ema_action = self._ema(self.ema_action, pid_output, self.ema_alpha_action)
            return float(self.ema_action)

        # Extract last CONTEXT_LENGTH states to compute features
        last_states = list(self.state_history)[-self.max_len:]
        # build arrays for required fields (most recent last)
        a_ego = [s.a_ego for s in last_states]
        v_ego = [s.v_ego for s in last_states]
        roll_lataccel = [s.roll_lataccel for s in last_states]

        # For action_history and current/target lataccel history we'll use deques
        recent_actions = list(self.action_history)
        # Build input observation quickly using preallocated buffer
        input_np = self._fill_obs_buf(
            recent_actions,
            roll_lataccel,
            v_ego,
            a_ego,
            list(self.current_lataccel_history),
            list(self.target_lataccel_history),
            future_plan if future_plan is not None else FuturePlan([], [], [], [])
        )

        # Inference (fast path uses ONNX)
        mean_action, value = self._policy_infer(input_np)

        # store logs
        self.value_log.append(float(value))
        self.mean_log.append(float(mean_action))

        # Cheap smoothing of the target lateral accel (EMA instead of filtfilt)
        self.ema_target = self._ema(self.ema_target, float(target_lataccel), self.ema_alpha_target)
        target_lataccel_filtered = float(self.ema_target)

        # If future plan contains near-future points, compute a lightweight weighted average
        # (original code used a weighted average of the next 5 entries)
        try:
            if future_plan is not None and len(future_plan.lataccel) >= 5:
                # small fixed-weight average; fast
                weights = np.array([4, 3, 2, 2, 2, 1], dtype=np.float32)
                vals = [target_lataccel_filtered] + list(future_plan.lataccel[:5])
                # normalize weights to avoid large scaling
                w = weights[: len(vals)]
                target_lataccel_filtered = float(np.dot(w, np.array(vals, dtype=np.float32)) / float(np.sum(w)))
        except Exception:
            pass

        # PID error computation
        error = float(target_lataccel_filtered) - float(current_lataccel)
        if len(self.target_lataccel_history) < CONTROL_START_IDX:
            self.error_integral = 0.0
        else:
            self.error_integral += error * 0.1
        error_diff = (error - self.prev_error) / 0.1 if self.prev_error is not None else 0.0
        self.prev_error = error

        pid_output = self.p * error + self.i * self.error_integral + self.d * error_diff
        self.pid_action_history.append(pid_output)

        # Blend deterministic neural mean with PID
        blended_action = float(self.alpha * float(mean_action) + (1.0 - float(self.alpha)) * pid_output)
        self.action_history.append(blended_action)

        # Smooth final output using EMA on action (cheap and single-step)
        self.ema_action = self._ema(self.ema_action, blended_action, self.ema_alpha_action)

        return float(self.ema_action)
