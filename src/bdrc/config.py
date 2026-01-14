# src/bdrc/config.py

from dataclasses import dataclass


@dataclass
class CostConfig:
    # --- c1: 仅完成率缺口 ---
    beta_gap: float = 1.5219         
    rho_star: float = 0.905           # 平台 SLA 目标，可按场景调整

    # --- c2: 重优化预算权重 ---
    alpha_n: float = 0.1639           # 每次调用成本权重
    alpha_m: float = 0.0016           # 每次改派成本权重

    # --- 缩放 ---
    reward_scale: float = 1.0 / 100.0
    c1_scale: float = 1.0 / 1000.0
    c2_scale: float = 1.0 / 100.0

    # --- c2 预算平滑（仅作方差抑制，可设 1.0 表示禁用额外平滑）---
    xi: float = 1.0

    # --- 防 freq→0 塌缩：最慢触发间隔 ---
    max_reopt_interval_sec: int = 1800


@dataclass
class LagrangeConfig:
    enabled: bool = True
    update_mu: bool = True
    budget_mode: str = "ema"  # ema=滑动平均更新; hard=使用当前批次均值

    lambda_init_c1: float = 0.3
    lambda_init_c2: float = 0.1
    lambda_lr_c1: float = 5e-3
    lambda_lr_c2: float = 2e-3

    # 预算：配合 scale 后的“单步平均”更合理
    cost_budget_c1: float = 0.25  
    cost_budget_c2: float = 0.3     # 结合 alpha_n/alpha_m 权重的单步期望预算
    cost_ema_beta: float = 0.05

    lambda_update_clip: float = 1.0
    lambda_max: float = 1e3

