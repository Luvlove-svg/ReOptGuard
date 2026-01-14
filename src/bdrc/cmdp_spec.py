# src/bdrc/cmdp_spec.py
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Set, Optional, List

import numpy as np

from src.simulation.discrete_event_env import DiscreteEventEnvironment
from src.simulation.state import SimulationState, OrderStatus, RiderStatus
from typing import Any, Dict, Tuple, Set, Optional
from src.bdrc.config import CostConfig

# ========= State 提取 =========

class BDRCStateExtractor:

    def __init__(self, env: DiscreteEventEnvironment):
        self.env = env
        self.total_orders = max(1, len(env.orders_data))
        self.total_riders = max(1, len(env.riders_data))
        self.start_time = getattr(env, "start_time", 0)
        self.end_time = getattr(env, "end_time", self.start_time + 1)
        self.max_orders_per_rider = env.constraints.get(
            "max_orders_per_rider", 5
        )
        # 用仿真总时长作为“总事件规模”的粗略近似
        self._horizon_sec = max(1, self.end_time - self.start_time)
        # 新增：缓存系统
        self._cached_state = None
        self._cached_time = -1
        self._cached_order_hash = 0
        self._cached_rider_hash = 0
        # 改进缓存：使用更有效的哈希
        self._last_state = None
        self._last_state_hash = 0
        self._cache_hits = 0
        self._cache_misses = 0
         # 预计算归一化因子
        self._norm_orders = 1.0 / max(1.0, float(self.total_orders))
        self._norm_riders = 1.0 / max(1.0, float(self.total_riders))
        self._norm_horizon = 1.0 / max(1.0, float(self._horizon_sec))
        self._norm_max_orders = 1.0 / max(1, self.max_orders_per_rider)
        
        if self.end_time > self.start_time:
            self._time_range = self.end_time - self.start_time
            self._norm_time = 1.0 / float(self._time_range)
        else:
            self._time_range = 1.0
            self._norm_time = 1.0

    def _compute_state_hash(self, state: SimulationState) -> int:
        """高效计算状态哈希，用于缓存检查"""
        # ✅兼容：state.riders 可能是 dict 或 list
        raw_riders = getattr(state, "riders", {})
        if isinstance(raw_riders, dict):
            rider_vals = raw_riders.values()
        else:
            try:
                rider_vals = list(raw_riders)
            except TypeError:
                rider_vals = []

        # 使用订单统计和骑手负载的组合哈希
        total_load = 0
        idle_cnt = 0
        for r in rider_vals:
            load = len(getattr(r, "assigned_orders", []))
            total_load += load
            if load == 0:
                idle_cnt += 1

        stats = (
            len(getattr(state, "completed_orders", [])),
            len(getattr(state, "cancelled_orders", [])),
            total_load,
            idle_cnt,
        )
        return hash(stats)

    def _compute_order_hash(self, state: SimulationState) -> int:
        """快速计算订单状态变化的哈希值"""
        # 使用完成和取消订单数量作为简单哈希
        return hash((len(state.completed_orders), len(state.cancelled_orders)))


    def _compute_rider_hash(self, state: SimulationState) -> int:
        """快速计算骑手状态变化的哈希值"""
        total_load = 0
        idle_count = 0
        for r in state.riders.values():
            total_load += len(r.assigned_orders)
            if len(r.assigned_orders) == 0:
                idle_count += 1
        return hash((total_load, idle_count))
    

    def extract(self, state: SimulationState) -> np.ndarray:
        now = state.current_time
        
        # 检查缓存
        state_hash = self._compute_state_hash(state)
        if self._last_state is not None and state_hash == self._last_state_hash:
            self._cache_hits += 1
            return self._last_state.copy()
        
        self._cache_misses += 1
        
        # 优化1：批量处理订单状态统计
        pending = 0
        in_progress = 0
        completed = len(state.completed_orders)
        cancelled = len(state.cancelled_orders)
        late_orders = 0
        
        # 优化：提前获取引用，减少属性访问
        raw_orders = getattr(state, "orders", {})
        raw_orders_data = getattr(self.env, "orders_data", {})

        if isinstance(raw_orders, dict):
            orders_items = raw_orders.items()
            orders_get = raw_orders.get
        else:
            # list/tuple: 构造轻量映射（仅本次 extract 使用）
            tmp_orders = {}
            try:
                for idx, o in enumerate(raw_orders):
                    oid = getattr(o, "order_id", None) or getattr(o, "id", None) or idx
                    tmp_orders[oid] = o
            except TypeError:
                tmp_orders = {}
            orders_items = tmp_orders.items()
            orders_get = tmp_orders.get

        if isinstance(raw_orders_data, dict):
            orders_data_get = raw_orders_data.get
        else:
            tmp_od = {}
            try:
                for idx, o in enumerate(raw_orders_data):
                    oid = getattr(o, "order_id", None) or getattr(o, "id", None) or idx
                    tmp_od[oid] = o
            except TypeError:
                tmp_od = {}
            orders_data_get = tmp_od.get

        # 使用 items() 避免二次查找
        for order_id, order_state in orders_items:
            status = order_state.status

            if status == OrderStatus.PENDING:
                pending += 1
            elif status in (OrderStatus.ASSIGNED, OrderStatus.PICKED_UP):
                in_progress += 1
                # 优化：如果大部分订单不会晚点，可以先检查状态
                order_data = orders_data_get(order_id)
                if order_data is not None:
                    if order_data.due_ts is not None and now > order_data.due_ts:
                        late_orders += 1
        # 优化骑手统计
        raw_riders = getattr(state, "riders", {})

        if isinstance(raw_riders, dict):
            rider_values = raw_riders.values()
        else:
            try:
                rider_values = list(raw_riders)
            except TypeError:
                rider_values = []

        active_riders = len(raw_riders) if isinstance(raw_riders, dict) else len(rider_values)
        idle_riders = 0
        total_load = 0

        # 使用 values() 和局部变量
        # 直接使用上面准备好的 rider_values，避免引用未定义变量
        for rider in rider_values:
            load = len(getattr(rider, "assigned_orders", []))
            total_load += load
            if load == 0:
                idle_riders += 1
        
        # 缓存骑手负载用于快速检查
        self._last_rider_load = total_load
        self._last_completed = completed
        self._last_cancelled = cancelled

        avg_rider_load = total_load / active_riders if active_riders > 0 else 0.0
        active_orders = pending + in_progress
        system_load = (
            active_orders / active_riders if active_riders > 0 else 0.0
        )

        # 获取重优化统计（修复1）
        stats = getattr(self.env, 'reopt_stats', {})
        reopt_calls = int(stats.get("reopt_calls", 0))
        reassigned_orders = int(stats.get("reassigned_orders", 0))

        # 使用预计算因子（避免每次计算除法）
        time_norm = 0.0
        if self._time_range > 0:
            time_norm = (now - self.start_time) * self._norm_time
        
        pending_ratio = pending * self._norm_orders
        in_progress_ratio = in_progress * self._norm_orders
        completed_ratio = completed * self._norm_orders
        cancelled_ratio = cancelled * self._norm_orders
        late_ratio = late_orders * self._norm_orders
        
        active_riders_ratio = active_riders * self._norm_riders
        idle_riders_ratio = idle_riders * self._norm_riders
        avg_load_norm = avg_rider_load * self._norm_max_orders
        
        reopt_calls_norm = reopt_calls * self._norm_horizon
        reassigned_orders_norm = reassigned_orders * self._norm_orders
        vec = np.array(
            [
                time_norm,
                pending_ratio,
                in_progress_ratio,
                completed_ratio,
                cancelled_ratio,
                late_ratio,
                active_riders_ratio,
                idle_riders_ratio,
                avg_load_norm,
                system_load,
                reopt_calls_norm,
                reassigned_orders_norm,
            ],
            dtype=np.float32,
        )
        # 更新缓存
        self._last_state = vec.copy()
        self._last_state_hash = state_hash
        
        return vec
    
    def extract_batch(self, states: List[SimulationState]) -> np.ndarray:
        """批量提取状态向量（如果需要）"""
        batch_size = len(states)
        vecs = np.zeros((batch_size, 12), dtype=np.float32)
        
        for i, state in enumerate(states):
            vecs[i] = self.extract(state)
        
        return vecs


# ========= Action 定义 & 映射 =========


@dataclass
class BDRCAction:
    __slots__ = ['reopt_freq_hz', 'max_reassign_per_round', 'min_cost_saving_threshold']
    reopt_freq_hz: float
    max_reassign_per_round: int
    min_cost_saving_threshold: float

    def clipped(self) -> "BDRCAction":
        FREQ_MAX_HZ = 1.0 / 30.0
        freq = float(np.clip(self.reopt_freq_hz, 0.0, FREQ_MAX_HZ))
        max_reassign = int(np.clip(self.max_reassign_per_round, 0, 1000))
        thr = float(np.clip(self.min_cost_saving_threshold, 0.0, 1e6))
        return BDRCAction(freq, max_reassign, thr)


class BDRCActionApplier:
    def __init__(self, env: DiscreteEventEnvironment):
        self.env = env

    def apply(self, action: BDRCAction) -> None:
        freq_hz = float(getattr(action, "reopt_freq_hz", 0.0) or 0.0)
        if freq_hz > 1e-6:
            interval_sec = int(max(1, 1.0 / freq_hz))
        else:
            interval_sec = 99999  # 频率接近0，几乎不重优化
        
        max_reassign = int(getattr(action, "max_reassign_per_round", 0) or 0)
        tau = float(getattr(action, "min_cost_saving_threshold", 0.0) or 0.0)

        # Env attribute used by `_maybe_reoptimize`
        if hasattr(self.env, "reopt_interval"):
            try:
                self.env.reopt_interval = interval_sec
            except Exception:
                pass

        # Config as source-of-truth
        if isinstance(getattr(self.env, "config", None), dict):
            reo_cfg = self.env.config.setdefault("reoptimization", {})
            reo_cfg["enabled"] = True
            reo_cfg["interval_sec"] = max(1, interval_sec)
            reo_cfg["max_reassign_per_round"] = max(0, max_reassign)
            reo_cfg["min_cost_saving_threshold"] = float(tau)

        # Live reoptimizer
        reo = getattr(self.env, "reoptimizer", None)
        if reo is not None:
            if hasattr(reo, "min_cost_saving"):
                try:
                    reo.min_cost_saving = float(tau)
                except Exception:
                    pass
            if hasattr(reo, "max_reassign_per_round"):
                try:
                    reo.max_reassign_per_round = max(0, int(max_reassign))
                except Exception:
                    pass
            if hasattr(reo, "interval_sec"):
                try:
                    reo.interval_sec = max(1, int(interval_sec))
                except Exception:
                    pass




class RewardCostCalculator:
    def __init__(self, env: DiscreteEventEnvironment, cost_cfg: Optional[CostConfig] = None):
        self.env = env
        self.orders_data = env.orders_data
        self.total_orders = max(1, len(self.orders_data))
        self._known_completed = set()
        self._prev_reopt_calls = 0
        self._prev_reassigned_orders = 0

        # 完成率缺口（论文固定公式）
        self.beta_gap = cost_cfg.beta_gap if cost_cfg else 1.5219
        self.rho_star = cost_cfg.rho_star if cost_cfg else 0.905
        # 重优化预算权重（论文固定公式）
        self.alpha_n = cost_cfg.alpha_n if cost_cfg else 0.1639
        self.alpha_m = cost_cfg.alpha_m if cost_cfg else 0.0016
        # 可选平滑（xi=1 等价于无平滑）
        self.xi = cost_cfg.xi if cost_cfg else 1.0
        self._b = 0.0  # 当前平滑预算占用
        self._completed_order_times = {}  # order_id -> (delivery_time, due_ts)
        self._last_known_completed_count = 0
        # 预计算缩放因子（仅稳定训练，用 config 覆盖）
        self._reward_scale = cost_cfg.reward_scale if cost_cfg else 1.0 / 100.0
        self._c1_scale = cost_cfg.c1_scale if cost_cfg else 1.0 / 1000.0
        self._c2_scale = cost_cfg.c2_scale if cost_cfg else 1.0 / 100.0
        
        # 预计算rho_star相关
        self._rho_star = self.rho_star


    def compute(self, state: SimulationState) -> Tuple[float, float, float]:
        # ---------- reward: 本步新完成订单数 ----------
        current_completed_count = len(state.completed_orders)
        new_completed_count = max(0, current_completed_count - self._last_known_completed_count)
        self._last_known_completed_count = current_completed_count
        reward = float(new_completed_count)

        # ---------- c1: 完成率缺口（固定公式） ----------
        rho_t = len(state.completed_orders) / float(self.total_orders)  # 当前总完成率
        gap = max(0.0, self.rho_star - rho_t)
        c1 = float(self.beta_gap * gap)

        # ---------- c2: 调用/改派的线性预算 ----------
        stats = self.env.reopt_stats or {}
        reopt_calls_total = int(stats.get("reopt_calls", 0))
        reassigned_total = int(stats.get("reassigned_orders", 0))

        delta_calls = max(0, reopt_calls_total - self._prev_reopt_calls)
        delta_reassign = max(0, reassigned_total - self._prev_reassigned_orders)
        self._prev_reopt_calls = reopt_calls_total
        self._prev_reassigned_orders = reassigned_total

        instant_usage = self.alpha_n * float(delta_calls) + self.alpha_m * float(delta_reassign)
        # 可选 EMA 平滑（xi=1 表示无平滑）
        self._b = (1.0 - self.xi) * self._b + self.xi * instant_usage
        c2 = self._b

        # 统一缩放（使用预计算因子）
        reward = reward * self._reward_scale
        c1 = c1 * self._c1_scale
        c2 = c2 * self._c2_scale

        return float(reward), float(c1), float(c2)

