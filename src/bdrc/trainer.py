# src/bdrc/trainer.py
"""BDRC Trainer with GPU acceleration and optimizations."""

from dataclasses import dataclass
from typing import Dict, Any
import logging
import numpy as np
import torch

from src.bdrc.cmdp_spec import (
    BDRCStateExtractor,
    BDRCActionApplier,
    RewardCostCalculator,
)
from src.bdrc.episode_runner import build_env
from src.bdrc.agent import BDRCAgent, AgentConfig
from src.bdrc.policy import PolicyConfig
from src.bdrc.critic import CriticConfig
from src.bdrc.buffer import OnPolicyBuffer

logger = logging.getLogger(__name__)

@dataclass
class TrainerConfig:
    max_decisions: int = 50
    decision_interval_sec: int = 300
    max_env_steps_per_episode: int = 200_000
    device: str = "cpu"
    # 新增：GPU加速相关
    use_gpu: bool = True
    gpu_batch_size: int = 32
    profile_performance: bool = False

class BDRCTrainer:
    def __init__(
        self,
        env_config: Dict[str, Any],
        data_dir: str,
        trainer_cfg: TrainerConfig,
        agent_cfg: AgentConfig,
        policy_cfg: PolicyConfig,
        critic_cfg: CriticConfig,
    ):
        self.env_config = env_config
        self.data_dir = data_dir
        self.trainer_cfg = trainer_cfg
        
        # 自动检测并设置设备
        device = self._auto_detect_device(trainer_cfg.device, trainer_cfg.use_gpu)
        
        # 更新设备配置
        agent_cfg.device = device
        trainer_cfg.device = device
        
        self.agent = BDRCAgent(
            agent_cfg, policy_cfg=policy_cfg, critic_cfg=critic_cfg, device=device
        )
        self.buffer = OnPolicyBuffer(state_dim=agent_cfg.state_dim)
        
        # 性能统计
        self._state_extract_time = 0
        self._env_step_time = 0
        self._agent_time = 0
        self._step_count = 0
        
        logger.info(f"训练器初始化完成，使用设备: {device}")
    
    def _auto_detect_device(self, device_str: str, use_gpu: bool) -> str:
        """自动检测并选择最佳设备"""
        if device_str == "auto":
            if use_gpu and torch.cuda.is_available():
                # 选择可用GPU
                num_gpus = torch.cuda.device_count()
                if num_gpus > 1:
                    # 可以选择最快的GPU
                    gpu_id = 0
                    logger.info(f"检测到 {num_gpus} 个GPU，使用GPU:{gpu_id}")
                    return f"cuda:{gpu_id}"
                else:
                    logger.info("检测到1个GPU，启用CUDA加速")
                    return "cuda"
            else:
                logger.info("使用CPU进行训练")
                return "cpu"
        return device_str
    
    def _timed_call(self, func, *args, **kwargs):
        """计时调用函数，返回(结果, 耗时)"""
        import time
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    def run_single_episode(self) -> dict:
        """优化后的episode运行"""
        env, orders_data, riders_data, restaurants_data = build_env(
            self.env_config, self.data_dir
        )

        # 防御性修复：确保 env.state 的 orders/riders/restaurants 为 dict
        from src.simulation.state import normalize_state
        try:
            normalize_state(env.state)
        except Exception as e:
            logger.warning(f"归一化仿真状态失败: {e}")

        state_extractor = BDRCStateExtractor(env)
        action_applier = BDRCActionApplier(env)
        rc_calc = RewardCostCalculator(env)

        self.buffer.reset()
        
        # 重置性能统计
        self._state_extract_time = 0
        self._env_step_time = 0
        self._agent_time = 0
        self._step_count = 0

        total_env_steps = 0
        decision_count = 0
        episode_reward_sum = 0.0
        episode_cost_c1_sum = 0.0
        episode_cost_c2_sum = 0.0
        
        # 预分配列表，减少动态扩展开销
        states_buffer = []
        actions_buffer = []
        log_probs_buffer = []
        values_buffer = []
        rewards_buffer = []
        costs_c1_buffer = []
        costs_c2_buffer = []
        dones_buffer = []

        while decision_count < self.trainer_cfg.max_decisions and env.running:
            sim_state = env.state
            now = sim_state.current_time

            # 1) 状态提取（已缓存优化）
            if self.trainer_cfg.profile_performance:
                s_vec, extract_time = self._timed_call(state_extractor.extract, sim_state)
                self._state_extract_time += extract_time
            else:
                s_vec = state_extractor.extract(sim_state)

            # 2) Agent选择动作
            if self.trainer_cfg.profile_performance:
                action_result, agent_time = self._timed_call(self.agent.select_action, s_vec)
                self._agent_time += agent_time
            else:
                action_result = self.agent.select_action(s_vec)
            
            # 拆包结果
            action, log_prob, value, raw_action_np = action_result
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[决策 {decision_count}] 动作: {action}")

            # 3) 应用动作到env
            action_applier.apply(action)

            # 4) 推进仿真到下一个决策时刻
            target_time = now + self.trainer_cfg.decision_interval_sec
            env_steps_in_interval = 0
            
            while env.running and env.state.current_time < target_time:
                if self.trainer_cfg.profile_performance:
                    _, step_time = self._timed_call(env.step)
                    self._env_step_time += step_time
                else:
                    env.step()
                
                total_env_steps += 1
                env_steps_in_interval += 1
                
                if total_env_steps >= self.trainer_cfg.max_env_steps_per_episode:
                    logger.warning(
                        "达到max_env_steps_per_episode=%d，提前结束episode",
                        self.trainer_cfg.max_env_steps_per_episode,
                    )
                    break

            # 5) 计算奖励/成本
            r_t, c1_t, c2_t = rc_calc.compute(env.state)
            episode_reward_sum += r_t
            episode_cost_c1_sum += c1_t
            episode_cost_c2_sum += c2_t
            done = not env.running

            # 6) 收集到缓冲区（先存在内存列表）
            states_buffer.append(s_vec)
            actions_buffer.append(raw_action_np)
            log_probs_buffer.append(log_prob)
            values_buffer.append(value)
            rewards_buffer.append(r_t)
            costs_c1_buffer.append(c1_t)
            costs_c2_buffer.append(c2_t)
            dones_buffer.append(done)

            decision_count += 1
            self._step_count += 1
            
            if total_env_steps >= self.trainer_cfg.max_env_steps_per_episode:
                break
        
        # 批量写入buffer，减少方法调用开销
        for i in range(len(states_buffer)):
            self.buffer.add(
                state_vec=states_buffer[i],
                action_raw=actions_buffer[i],
                log_prob=log_probs_buffer[i],
                value=values_buffer[i],
                reward=rewards_buffer[i],
                cost_c1=costs_c1_buffer[i],
                cost_c2=costs_c2_buffer[i],
                done=dones_buffer[i],
            )

        # episode循环结束
        if len(self.buffer) == 0:
            logger.warning("本episode中没有任何决策步，跳过agent.update()")
            stats_update = {
                "loss_total": 0.0,
                "loss_policy": 0.0,
                "loss_value": 0.0,
                "entropy": 0.0,
                "returns_mean": 0.0,
                "lambda_c1": float(getattr(self.agent, "lambda_c1", 0.0)),
                "lambda_c2": float(getattr(self.agent, "lambda_c2", 0.0)),
            }
        else:
            batch = self.buffer.get_batch()
            stats_update = self.agent.update(batch)
        
        # 添加性能统计
        if self.trainer_cfg.profile_performance and self._step_count > 0:
            avg_state_time = self._state_extract_time / self._step_count
            avg_step_time = self._env_step_time / total_env_steps if total_env_steps > 0 else 0
            avg_agent_time = self._agent_time / self._step_count
            
            logger.info(
                f"性能统计: 状态提取={avg_state_time:.4f}s/步, "
                f"环境步进={avg_step_time:.4f}s/步, "
                f"智能体={avg_agent_time:.4f}s/步"
            )
            
            stats_update.update({
                "avg_state_extract_time": avg_state_time,
                "avg_env_step_time": avg_step_time,
                "avg_agent_time": avg_agent_time,
            })

        result = {
            "episode_reward_sum": episode_reward_sum,
            "episode_cost_c1_sum": episode_cost_c1_sum,
            "episode_cost_c2_sum": episode_cost_c2_sum,
            "episode_decisions": decision_count,
            "episode_env_steps": total_env_steps,
            "reopt_stats": getattr(env, "reopt_stats", None),
        }
        result.update(stats_update)
        return result
    
    def get_agent_state(self) -> dict:
        return self.agent.to_state_dict()
    
    def load_agent_state(self, state: dict) -> None:
        self.agent.load_from_state_dict(state)