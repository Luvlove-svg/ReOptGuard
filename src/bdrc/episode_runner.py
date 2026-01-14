from typing import Callable, Dict, Any, List, Tuple
import os
import logging
import copy
import json
import hashlib

import numpy as np

from src.simulation.discrete_event_env import DiscreteEventEnvironment
from src.data.loaders import DataLoader
from src.bdrc.cmdp_spec import (
    BDRCStateExtractor,
    BDRCAction,
    BDRCActionApplier,
    RewardCostCalculator,
)
from src.simulation.state import SimulationState

logger = logging.getLogger(__name__)

PolicyFn = Callable[[np.ndarray], BDRCAction]

# 全局缓存
_DATA_CACHE = {}
_ENV_CACHE = {}

def _get_data_hash(data_dir: str) -> str:
    """计算数据目录的哈希值，用于缓存键"""
    files = ["orders.json", "riders.json", "restaurants.json"]
    hash_input = ""
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            mtime = os.path.getmtime(file_path)
            size = os.path.getsize(file_path)
            hash_input += f"{file_name}:{mtime}:{size}:"
    return hashlib.md5(hash_input.encode()).hexdigest()

def _load_data_cached(data_dir: str):
    cache_key = _get_data_hash(data_dir)
    
    if cache_key in _DATA_CACHE:
        logger.debug(f"从缓存加载数据: {data_dir}")
        cached_data = _DATA_CACHE[cache_key]
        
        # 确保返回的是字典
        if isinstance(cached_data, tuple) and len(cached_data) == 3:
            orders_data, riders_data, restaurants_data = cached_data
            
            # 修复：确保缓存中存储的是字典，如果不是就转换并更新缓存
            needs_update = False
            
            if not isinstance(orders_data, dict):
                logger.warning(f"缓存中的 orders_data 类型是 {type(orders_data)}，转换为字典")
                orders_data = {o.order_id: o for o in orders_data}
                needs_update = True
            if not isinstance(riders_data, dict):
                logger.warning(f"缓存中的 riders_data 类型是 {type(riders_data)}，转换为字典")
                riders_data = {r.rider_id: r for r in riders_data}
                needs_update = True
            if not isinstance(restaurants_data, dict):
                logger.warning(f"缓存中的 restaurants_data 类型是 {type(restaurants_data)}，转换为字典")
                restaurants_data = {r.restaurant_id: r for r in restaurants_data}
                needs_update = True
            
            # 如果转换了，更新缓存
            if needs_update:
                _DATA_CACHE[cache_key] = (orders_data, riders_data, restaurants_data)
                logger.debug(f"更新缓存中的数据为字典格式")
                
            # 返回深拷贝，确保每个环境独立
            return (copy.deepcopy(orders_data), 
                    copy.deepcopy(riders_data), 
                    copy.deepcopy(restaurants_data))
        else:
            # 如果缓存数据格式不对，重新加载
            logger.warning(f"缓存数据格式错误: {type(cached_data)}，重新加载")
    
    # 原始加载逻辑
    orders_file = os.path.join(data_dir, "orders.json")
    riders_file = os.path.join(data_dir, "riders.json")
    restaurants_file = os.path.join(data_dir, "restaurants.json")

    orders_list = DataLoader.load_orders(orders_file)
    riders_list = DataLoader.load_riders(riders_file)
    restaurants_list = DataLoader.load_restaurants(restaurants_file)

    orders_dict = {o.order_id: o for o in orders_list}
    riders_dict = {r.rider_id: r for r in riders_list}
    restaurants_dict = {rst.restaurant_id: rst for rst in restaurants_list}
    
    # 缓存原始数据（确保是字典）
    _DATA_CACHE[cache_key] = (orders_dict, riders_dict, restaurants_dict)
    
    logger.info(f"加载并缓存数据: {data_dir}, 缓存键: {cache_key}")
    return orders_dict, riders_dict, restaurants_dict



def _precompute_time_ranges(orders_data, riders_data):
    """
    预计算时间范围 - 修复版本，确保正确处理所有情况
    """
    # 辅助函数：获取订单时间列表
    def get_order_times(data):
        if isinstance(data, dict):
            return [o.ready_ts for o in data.values()], [o.due_ts for o in data.values()]
        elif isinstance(data, list):
            return [o.ready_ts for o in data], [o.due_ts for o in data]
        else:
            raise TypeError(f"orders_data 必须是字典或列表，当前是 {type(data)}")
    
    # 辅助函数：获取骑手时间列表
    def get_rider_times(data):
        if isinstance(data, dict):
            return [r.shift_start for r in data.values()], [r.shift_end for r in data.values()]
        elif isinstance(data, list):
            return [r.shift_start for r in data], [r.shift_end for r in data]
        else:
            raise TypeError(f"riders_data 必须是字典或列表，当前是 {type(data)}")
    
    order_times, order_due_times = get_order_times(orders_data)
    rider_shift_starts, rider_shift_ends = get_rider_times(riders_data)
    
    start_time = min(order_times + rider_shift_starts)
    end_time = max(order_due_times + rider_shift_ends)
    
    return start_time, end_time


def build_env(config: Dict[str, Any], data_dir: str) -> Tuple[
    DiscreteEventEnvironment,
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
]:
    # 使用带缓存的加载
    orders_data, riders_data, restaurants_data = _load_data_cached(data_dir)

    logger.info(
        "BDRC: 加载数据: %d 个订单, %d 个骑手",
        len(orders_data),
        len(riders_data),
    )

    # 预计算时间范围
    start_time, end_time = _precompute_time_ranges(orders_data, riders_data)

    # 修复：正确获取订单时间范围（处理字典和列表两种情况）
    def get_min_max_order_times(data):
        if isinstance(data, dict):
            order_times = [o.ready_ts for o in data.values()]
        elif isinstance(data, list):
            order_times = [o.ready_ts for o in data]
        else:
            return 0, 0
        if not order_times:
            logger.warning("订单数据为空，使用默认时间范围")
            return 0, 0
        return min(order_times), max(order_times)
    
    min_order_time, max_order_time = get_min_max_order_times(orders_data)
    logger.info(
        "BDRC: 订单 ready_ts 范围: %s - %s",
        min_order_time,
        max_order_time,
    )

    # 修复：正确获取骑手时间范围（处理字典和列表两种情况）
    def get_min_max_rider_times(data):
        if isinstance(data, dict):
            rider_times = [r.shift_start for r in data.values()]
        elif isinstance(data, list):
            rider_times = [r.shift_start for r in data]
        else:
            return 0, 0
        if not rider_times:
            logger.warning("骑手数据为空，使用默认时间范围")
            return 0, 0
        return min(rider_times), max(rider_times)
    
    min_rider_time, max_rider_time = get_min_max_rider_times(riders_data)
    logger.info(
        "BDRC: 骑手上班时间范围: %s - %s",
        min_rider_time,
        max_rider_time,
    )

    env = DiscreteEventEnvironment(config, orders_data, riders_data, restaurants_data)

    # 设置时间范围
    env.start_time = start_time
    env.end_time = end_time

    logger.info(
        "BDRC: 仿真时间范围: %s - %s (总时长 %s 秒)",
        env.start_time,
        env.end_time,
        env.end_time - env.start_time,
    )

    env.reset()  # 设置好事件队列等
    return env, orders_data, riders_data, restaurants_data


def build_env_fast(config: Dict[str, Any], data_dir: str, reset_only: bool = False) -> Tuple[
    DiscreteEventEnvironment,
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
]:
    # 生成缓存键：配置 + 数据目录
    config_str = json.dumps(config, sort_keys=True)
    cache_key = hashlib.md5((config_str + data_dir).encode()).hexdigest()
    
    # 如果 reset_only 为 True，尝试使用缓存
    if reset_only and cache_key in _ENV_CACHE:
        logger.debug(f"使用缓存的环境原型: {data_dir}")
        
        # 从缓存获取原型
        cached_env, orders_data, riders_data, restaurants_data = _ENV_CACHE[cache_key]
        
        try:
            # 尝试复制环境
            new_env = copy.deepcopy(cached_env)
            new_env.reset()  # 重置到初始状态
            
            # 复制数据（确保独立）
            orders_copy = copy.deepcopy(orders_data)
            riders_copy = copy.deepcopy(riders_data)
            restaurants_copy = copy.deepcopy(restaurants_data)
            
            # 重新关联数据
            new_env.orders_data = orders_copy
            new_env.riders_data = riders_copy
            new_env.restaurants_data = restaurants_copy
            
            # 重新计算时间范围（如果需要）
            if hasattr(new_env, 'start_time') and hasattr(new_env, 'end_time'):
                order_times = [o.ready_ts for o in orders_copy.values()]
                rider_shift_starts = [r.shift_start for r in riders_copy.values()]
                new_env.start_time = min(order_times + rider_shift_starts)
                new_env.end_time = max(
                    [o.due_ts for o in orders_copy.values()]
                    + [r.shift_end for r in riders_copy.values()]
                )
            
            logger.debug(f"从缓存成功复制并重置环境")
            return new_env, orders_copy, riders_copy, restaurants_copy
            
        except Exception as e:
            logger.warning(f"环境复制失败，回退到新建: {e}")
            # 回退到正常构建
            pass
    
    # 正常构建（第一次或复制失败）
    logger.debug(f"新建环境: {data_dir}")
    env, orders_data, riders_data, restaurants_data = build_env(config, data_dir)
    
    # 缓存原型
    _ENV_CACHE[cache_key] = (env, orders_data, riders_data, restaurants_data)
    
    return env, orders_data, riders_data, restaurants_data


def run_bdrc_episode(
    config: Dict[str, Any],
    data_dir: str,
    policy_fn: PolicyFn,
    max_decisions: int = 50,
    decision_interval_sec: int = 300,
    max_total_env_steps: int = 200_000,
    use_fast_build: bool = False,
) -> Dict[str, Any]:

    # 根据参数选择构建方式
    if use_fast_build:
        env, orders_data, riders_data, restaurants_data = build_env_fast(config, data_dir)
    else:
        env, orders_data, riders_data, restaurants_data = build_env(config, data_dir)
        
    state_extractor = BDRCStateExtractor(env)
    action_applier = BDRCActionApplier(env)
    rc_calc = RewardCostCalculator(env)

    states: List[np.ndarray] = []
    actions: List[BDRCAction] = []
    rewards: List[float] = []
    costs_c1: List[float] = []
    costs_c2: List[float] = []
    times: List[int] = []

    total_env_steps = 0
    decision_idx = 0

    while decision_idx < max_decisions and env.running:
        sim_state: SimulationState = env.state
        now = sim_state.current_time

        # 1) 抽 state 向量
        s_vec = state_extractor.extract(sim_state)

        # 2) policy 输出动作
        action = policy_fn(s_vec)
        if not isinstance(action, BDRCAction):
            raise TypeError(
                f"policy_fn 必须返回 BDRCAction，当前返回类型: {type(action)}"
            )

        # 3) 应用动作到 env（修改 Reopt 参数）
        action_applier.apply(action)

        # 4) 推进底层仿真，直到下一个决策时刻
        target_time = now + decision_interval_sec
        while env.running and env.state.current_time < target_time:
            env.step()
            total_env_steps += 1
            if total_env_steps >= max_total_env_steps:
                logger.warning(
                    "BDRC: 达到 max_total_env_steps=%d，提前结束 episode",
                    max_total_env_steps,
                )
                break

        # 5) 计算此决策间隔内的 reward / cost 增量
        r_t, c1_t, c2_t = rc_calc.compute(env.state)

        # 6) 记录
        states.append(s_vec)
        actions.append(action)
        rewards.append(r_t)
        costs_c1.append(c1_t)
        costs_c2.append(c2_t)
        times.append(env.state.current_time)

        decision_idx += 1

        if total_env_steps >= max_total_env_steps:
            break

    logger.info(
        "BDRC: episode 结束，决策步数=%d，总 env 步数=%d，reopt_stats=%s",
        len(states),
        total_env_steps,
        getattr(env, "reopt_stats", None),
    )

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "costs_c1": costs_c1,
        "costs_c2": costs_c2,
        "times": times,
        "env": env,
        "orders_data": orders_data,
        "riders_data": riders_data,
        "restaurants_data": restaurants_data,
        "reopt_stats": getattr(env, "reopt_stats", None),
    }