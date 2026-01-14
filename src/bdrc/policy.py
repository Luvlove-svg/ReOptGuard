# src/bdrc/policy.py

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from src.bdrc.cmdp_spec import BDRCAction


@dataclass
class PolicyConfig:
    state_dim: int = 12
    # 原先的 hidden_sizes，為兼容保留；若 mono/free 未設置，則沿用此配置。
    hidden_sizes: Sequence[int] = (64, 64)
    # 單調分支 / 自由分支的隱層寬度；若為 None，則使用 hidden_sizes。
    mono_hidden_sizes: Optional[Sequence[int]] = None
    free_hidden_sizes: Optional[Sequence[int]] = None
    max_reassign_per_round: int = 200
    max_cost_saving_threshold: float = 1e5
    log_std_init: float = -0.5  # 高斯策略初始對數標準差
    # 单动作消融：仅保留一个动作维度，其余置零
    single_action_only: bool = False
    single_action_index: int = 0
    # --- exploration / scaling knobs ---
    # Maximum re-optimization frequency (Hz). Default keeps original upper bound (1/30).
    reopt_freq_max_hz: float = 1.0 / 30.0
    # Target (initial) re-optimization interval in seconds (used to auto-compute a logit bias).
    reopt_target_interval_sec: float = 300.0
    # Optional explicit logit bias added to raw action[0] before sigmoid; if None, computed from target interval.
    reopt_freq_logit_bias: Optional[float] = None
    # 策略類型: "monotone" (帶單調結構) 或 "mlp" (普通 MLP 基線)
    policy_type: str = "monotone"

class NonNegativeLinear(nn.Module):
    """
    权重被约束为非负的线性层：
      y = x @ softplus(W_raw).T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.softplus(self.raw_weight)  # >= 0
        y = x @ weight.t()
        if self.bias is not None:
            y = y + self.bias
        return y



class BDRCPolicy(nn.Module):

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.policy_type = cfg.policy_type.lower()
        self.idx_l = [1, 2, 8, 9]                 # 負載特徵 ℓ
        self.idx_u = [5]                          # 晚單 / 風險特徵 u
        self.idx_v = [0, 3, 4, 6, 7, 10, 11]      # 其他上下文 v
        mono_sizes = list(cfg.mono_hidden_sizes or cfg.hidden_sizes)
        free_sizes = list(cfg.free_hidden_sizes or cfg.hidden_sizes)
        if self.policy_type == "monotone":
            # —— Monotone 分支：輸入 concat(ℓ, u)，權重非負 —— 
            in_mono = len(self.idx_l) + len(self.idx_u)
            mono_layers = []
            last_dim = in_mono
            for h in mono_sizes:
                mono_layers.append(NonNegativeLinear(last_dim, h))
                mono_layers.append(nn.ReLU())
                last_dim = h
            self.mono_backbone = nn.Sequential(*mono_layers)
            # 對 (f, k, tau) 輸出單調貢獻
            self.mono_head = NonNegativeLinear(last_dim, 3)
            # —— Free 分支：輸入 v，自由 MLP —— 
            in_free = len(self.idx_v)
            free_layers = []
            last_dim = in_free
            for h in free_sizes:
                free_layers.append(nn.Linear(last_dim, h))
                free_layers.append(nn.ReLU())
                last_dim = h
            self.free_backbone = nn.Sequential(*free_layers)
            self.free_head = nn.Linear(last_dim, 3)
        elif self.policy_type == "mlp":
            # 普通 MLP 策略：直接用整個 state 作為輸入
            mlp_sizes = list(cfg.hidden_sizes)
            in_dim = cfg.state_dim
            layers = []
            last_dim = in_dim
            for h in mlp_sizes:
                layers.append(nn.Linear(last_dim, h))
                layers.append(nn.ReLU())
                last_dim = h
            self.mlp_backbone = nn.Sequential(*layers)
            self.mlp_head = nn.Linear(last_dim, 3)
        else:
            raise ValueError(f"Unknown policy_type: {cfg.policy_type}")
        # —— 高斯策略的 log_std（3 維，共享） —— 
        self.log_std = nn.Parameter(
            torch.ones(3, dtype=torch.float32) * cfg.log_std_init
        )

        # 初始化權重為小值，避免 raw mean 飽和 sigmoid
        self._init_weights()

    def _init_weights(self) -> None:
        """Lightweight init to keep raw logits near 0 and avoid sigmoid saturation."""
        for m in self.modules():
            if isinstance(m, NonNegativeLinear):
                # Softplus(−4)≈0.018，保證初始權重極小，避免多層累乘後爆炸
                nn.init.normal_(m.raw_weight, mean=-4.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.policy_type == "monotone":
            x_l = states[:, self.idx_l]
            x_u = states[:, self.idx_u]
            x_v = states[:, self.idx_v]

            x_m = torch.cat([x_l, x_u], dim=-1)
            mono_feat = self.mono_backbone(x_m)
            mono_out = self.mono_head(mono_feat)
            free_feat = self.free_backbone(x_v)
            free_out = self.free_head(free_feat)
            mean = mono_out + free_out
        else:
            feat = self.mlp_backbone(states)
            mean = self.mlp_head(feat)
        return mean, self.log_std

    def _transform_to_action_space(self, raw: torch.Tensor) -> torch.Tensor:
        f_max = float(getattr(self.cfg, "reopt_freq_max_hz", 1.0 / 30.0))
        bias = getattr(self.cfg, "reopt_freq_logit_bias", None)
        if bias is None:
            target_interval = float(getattr(self.cfg, "reopt_target_interval_sec", 300.0))
            p = (1.0 / max(target_interval, 1e-9)) / max(f_max, 1e-9)
            p = min(max(p, 1e-3), 1.0 - 1e-3)
            import math
            bias = math.log(p / (1.0 - p))
        freq = torch.sigmoid(raw[..., 0] + float(bias)) * f_max

        # 1) max_reassign_per_round: [0, max_reassign_per_round]
        max_reassign = torch.sigmoid(raw[..., 1]) * float(
            self.cfg.max_reassign_per_round
        )

        # 2) min_cost_saving_threshold: [0, max_cost_saving_threshold]，對 raw 單調遞減
        #    base = sigmoid(raw_tau) in (0,1)，thr = (1 - base) * tau_max
        base = torch.sigmoid(raw[..., 2])
        thr = (1.0 - base) * float(self.cfg.max_cost_saving_threshold)
        thr = torch.clamp(thr, 0.0, float(self.cfg.max_cost_saving_threshold))

        return torch.stack([freq, max_reassign, thr], dim=-1)

    def distribution(self, states: torch.Tensor) -> D.Independent:
        mean, log_std = self(states)
        std = torch.exp(log_std)  # shape (3,)
        dist = D.Normal(mean, std)
        return D.Independent(dist, 1)  # 把 3 維視為一個 joint

    def sample_action(
        self, state_vec: np.ndarray
    ) -> Tuple[BDRCAction, torch.Tensor, torch.Tensor]:
        """
        numpy 版接口：内部转成 tensor，走 sample_action_tensor（更快、更稳）。
        """
        device = next(self.parameters()).device
        state_t = torch.as_tensor(state_vec, dtype=torch.float32, device=device)
        return self.sample_action_tensor(state_t)


    def sample_action_tensor(
        self, state_t: torch.Tensor
    ) -> Tuple[BDRCAction, torch.Tensor, torch.Tensor]:
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        state_t = state_t.to(next(self.parameters()).device, dtype=torch.float32)

        with torch.no_grad():
            dist = self.distribution(state_t)          # raw-space distribution
            raw_action_b = dist.sample()               # shape (1,3)
            log_prob_b = dist.log_prob(raw_action_b)   # shape (1,)

            raw_action = raw_action_b.squeeze(0).detach()

            # 单动作消融：仅保留一个动作维度，其余置零
            if self.cfg.single_action_only:
                idx = int(self.cfg.single_action_index)
                idx = max(0, min(2, idx))
                masked = torch.zeros_like(raw_action)
                masked[idx] = raw_action[idx]
                raw_action = masked
                log_prob_b = dist.log_prob(raw_action.unsqueeze(0))

            log_prob = log_prob_b.squeeze(0).detach()

            transformed = self._transform_to_action_space(raw_action)  # shape (3,)
            action = BDRCAction(
                reopt_freq_hz=float(transformed[0].item()),
                max_reassign_per_round=float(transformed[1].item()),
                min_cost_saving_threshold=float(transformed[2].item()),
            ).clipped()

        return action, log_prob, raw_action