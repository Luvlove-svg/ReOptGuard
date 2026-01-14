# src/bdrc/__init__.py
"""
BDRC: Budgeted Dynamic Reoptimization Control

本包实现：
- 将离散事件仿真环境映射为 CMDP (state / action / reward / costs)
- 后续会在此基础上实现策略网络、价值网络和 Lagrangian Actor–Critic
"""

from .cmdp_spec import (
    BDRCStateExtractor,
    BDRCAction,
    BDRCActionApplier,
    RewardCostCalculator,
)
